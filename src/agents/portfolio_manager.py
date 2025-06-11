# 导入JSON处理模块
import json
# 导入LangChain核心消息类
from langchain_core.messages import HumanMessage
# 导入LangChain聊天提示模板
from langchain_core.prompts import ChatPromptTemplate

# 导入代理状态和推理显示功能
from src.graph.state import AgentState, show_agent_reasoning
# 导入Pydantic数据模型
from pydantic import BaseModel, Field
# 导入类型注解
from typing_extensions import Literal
# 导入进度跟踪工具
from src.utils.progress import progress
# 导入大语言模型调用工具
from src.utils.llm import call_llm


class PortfolioDecision(BaseModel):
    """投资组合决策数据模型"""
    action: Literal["buy", "sell", "short", "cover", "hold"]  # 交易动作：买入、卖出、做空、平仓、持有
    quantity: int = Field(description="要交易的股票数量")  # 交易数量
    confidence: float = Field(description="决策的置信度，范围0.0到100.0")  # 置信度
    reasoning: str = Field(description="决策的推理过程")  # 推理说明


class PortfolioManagerOutput(BaseModel):
    """投资组合管理器输出数据模型"""
    decisions: dict[str, PortfolioDecision] = Field(description="股票代码到交易决策的字典映射")


##### 投资组合管理代理 #####
def portfolio_management_agent(state: AgentState):
    """为多个股票做出最终交易决策并生成订单
    
    Args:
        state: 包含投资组合、分析师信号和股票代码的代理状态
        
    Returns:
        dict: 更新后的状态，包含投资组合管理决策消息
    """

    # 获取投资组合和分析师信号
    portfolio = state["data"]["portfolio"]  # 当前投资组合状态
    analyst_signals = state["data"]["analyst_signals"]  # 所有分析师的信号
    tickers = state["data"]["tickers"]  # 要分析的股票代码列表

    # 为每个股票获取头寸限制、当前价格和信号
    position_limits = {}  # 头寸限制字典
    current_prices = {}   # 当前价格字典
    max_shares = {}       # 最大可交易股数字典
    signals_by_ticker = {} # 按股票分组的信号字典
    
    for ticker in tickers:
        # 更新进度状态
        progress.update_status("portfolio_manager", ticker, "处理分析师信号")

        # 从风险管理代理获取该股票的头寸限制和当前价格
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        position_limits[ticker] = risk_data.get("remaining_position_limit", 0)  # 剩余头寸限制
        current_prices[ticker] = risk_data.get("current_price", 0)  # 当前股价

        # 根据头寸限制和价格计算允许的最大股数
        if current_prices[ticker] > 0:
            max_shares[ticker] = int(position_limits[ticker] / current_prices[ticker])
        else:
            max_shares[ticker] = 0

        # 收集该股票的所有分析师信号
        ticker_signals = {}
        for agent, signals in analyst_signals.items():
            # 跳过风险管理代理，只收集其他分析师的信号
            if agent != "risk_management_agent" and ticker in signals:
                ticker_signals[agent] = {"signal": signals[ticker]["signal"], "confidence": signals[ticker]["confidence"]}
        signals_by_ticker[ticker] = ticker_signals

    # 更新进度状态为生成交易决策
    progress.update_status("portfolio_manager", None, "生成交易决策")

    # 生成交易决策
    result = generate_trading_decision(
        tickers=tickers,                    # 股票代码列表
        signals_by_ticker=signals_by_ticker, # 按股票分组的信号
        current_prices=current_prices,       # 当前价格
        max_shares=max_shares,              # 最大可交易股数
        portfolio=portfolio,                # 当前投资组合
        state=state,                        # 代理状态
    )

    # 创建投资组合管理消息
    message = HumanMessage(
        content=json.dumps({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}),
        name="portfolio_manager",
    )

    # 如果设置了显示推理标志，则打印决策
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning({ticker: decision.model_dump() for ticker, decision in result.decisions.items()}, "投资组合管理器")

    # 更新进度状态为完成
    progress.update_status("portfolio_manager", None, "完成")

    # 返回更新后的状态
    return {
        "messages": state["messages"] + [message],  # 添加新消息到消息列表
        "data": state["data"],                      # 保持数据不变
    }


def generate_trading_decision(
    tickers: list[str],
    signals_by_ticker: dict[str, dict],
    current_prices: dict[str, float],
    max_shares: dict[str, int],
    portfolio: dict[str, float],
    state: AgentState,
) -> PortfolioManagerOutput:
    """使用重试逻辑从大语言模型获取交易决策
    
    Args:
        tickers: 股票代码列表
        signals_by_ticker: 按股票分组的分析师信号字典
        current_prices: 当前价格字典
        max_shares: 最大可交易股数字典
        portfolio: 当前投资组合状态
        state: 代理状态
        
    Returns:
        PortfolioManagerOutput: 包含所有股票交易决策的输出对象
    """
    # 创建提示模板
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个投资组合管理器，基于多个股票代码做出最终交易决策。

              交易规则:
              - 对于多头头寸:
                * 只有在有可用现金时才能买入
                * 只有在当前持有该股票的多头股份时才能卖出
                * 卖出数量必须 ≤ 当前多头头寸股数
                * 买入数量必须 ≤ 该股票的最大股数限制
              
              - 对于空头头寸:
                * 只有在有可用保证金时才能做空（头寸价值 × 保证金要求）
                * 只有在当前持有该股票的空头股份时才能平仓
                * 平仓数量必须 ≤ 当前空头头寸股数
                * 做空数量必须遵守保证金要求
              
              - max_shares 值已预先计算以遵守头寸限制
              - 根据信号考虑多头和空头机会
              - 通过多头和空头敞口维持适当的风险管理

              可用操作:
              - "buy": 开仓或增加多头头寸
              - "sell": 平仓或减少多头头寸
              - "short": 开仓或增加空头头寸
              - "cover": 平仓或减少空头头寸
              - "hold": 无操作

              输入参数:
              - signals_by_ticker: 股票代码 → 信号的字典
              - max_shares: 每个股票允许的最大股数
              - portfolio_cash: 投资组合中的当前现金
              - portfolio_positions: 当前头寸（多头和空头）
              - current_prices: 每个股票的当前价格
              - margin_requirement: 空头头寸的当前保证金要求（例如，0.5表示50%）
              - total_margin_used: 当前使用的总保证金
              """,
            ),
            (
                "human",
                """基于团队的分析，为每个股票代码做出交易决策。

              以下是按股票分组的信号:
              {signals_by_ticker}

              当前价格:
              {current_prices}

              允许购买的最大股数:
              {max_shares}

              投资组合现金: {portfolio_cash}
              当前头寸: {portfolio_positions}
              当前保证金要求: {margin_requirement}
              已使用总保证金: {total_margin_used}

              严格按照以下JSON结构输出:
              {{
                "decisions": {{
                  "TICKER1": {{
                    "action": "buy/sell/short/cover/hold",
                    "quantity": integer,
                    "confidence": float between 0 and 100,
                    "reasoning": "string"
                  }},
                  "TICKER2": {{
                    ...
                  }},
                  ...
                }}
              }}
              """,
            ),
        ]
    )

    # 生成提示
    prompt = template.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),      # 按股票分组的信号（JSON格式）
            "current_prices": json.dumps(current_prices, indent=2),            # 当前价格（JSON格式）
            "max_shares": json.dumps(max_shares, indent=2),                    # 最大股数限制（JSON格式）
            "portfolio_cash": f"{portfolio.get('cash', 0):.2f}",              # 投资组合现金（格式化为两位小数）
            "portfolio_positions": json.dumps(portfolio.get("positions", {}), indent=2),  # 当前头寸（JSON格式）
            "margin_requirement": f"{portfolio.get('margin_requirement', 0):.2f}",       # 保证金要求（格式化为两位小数）
            "total_margin_used": f"{portfolio.get('margin_used', 0):.2f}",              # 已使用总保证金（格式化为两位小数）
        }
    )

    # 创建投资组合管理器输出的默认工厂函数
    def create_default_portfolio_output():
        """当LLM调用失败时，为所有股票创建默认的持有决策"""
        return PortfolioManagerOutput(
            decisions={
                ticker: PortfolioDecision(
                    action="hold",                                    # 默认动作：持有
                    quantity=0,                                       # 数量：0
                    confidence=0.0,                                   # 置信度：0
                    reasoning="投资组合管理出错，默认持有"                # 推理：错误说明
                ) for ticker in tickers
            }
        )

    # 调用大语言模型生成交易决策
    return call_llm(
        prompt=prompt,                                    # 生成的提示
        pydantic_model=PortfolioManagerOutput,          # 输出数据模型
        agent_name="portfolio_manager",                 # 代理名称
        state=state,                                     # 代理状态
        default_factory=create_default_portfolio_output, # 默认输出工厂函数
    )
