# 导入代理状态和推理显示功能
from src.graph.state import AgentState, show_agent_reasoning
# 导入LangChain聊天提示模板
from langchain_core.prompts import ChatPromptTemplate
# 导入LangChain核心消息类
from langchain_core.messages import HumanMessage
# 导入Pydantic数据模型
from pydantic import BaseModel
# 导入JSON处理模块
import json
# 导入类型注解
from typing_extensions import Literal
# 导入金融数据API工具
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
# 导入大语言模型调用工具
from src.utils.llm import call_llm
# 导入进度跟踪工具
from src.utils.progress import progress


class WarrenBuffettSignal(BaseModel):
    """巴菲特分析信号数据模型"""
    signal: Literal["bullish", "bearish", "neutral"]  # 信号类型：看涨、看跌、中性
    confidence: float  # 置信度
    reasoning: str     # 推理过程


def warren_buffett_agent(state: AgentState):
    """使用巴菲特投资原则和大语言模型推理分析股票
    
    Args:
        state: 包含股票代码和结束日期的代理状态
        
    Returns:
        dict: 更新后的状态，包含巴菲特分析结果消息
    """
    data = state["data"]
    end_date = data["end_date"]      # 分析结束日期
    tickers = data["tickers"]        # 要分析的股票代码列表

    # 收集所有分析数据用于大语言模型推理
    analysis_data = {}     # 存储分析数据
    buffett_analysis = {}  # 存储巴菲特分析结果

    for ticker in tickers:
        # 更新进度状态：获取财务指标
        progress.update_status("warren_buffett_agent", ticker, "获取财务指标")
        # 获取所需数据 - 请求更多期间以便更好地进行趋势分析
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)

        # 更新进度状态：收集财务科目
        progress.update_status("warren_buffett_agent", ticker, "收集财务科目")
        # 搜索特定的财务科目数据
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",                        # 资本支出
                "depreciation_and_amortization",              # 折旧和摊销
                "net_income",                                 # 净收入
                "outstanding_shares",                         # 流通股数
                "total_assets",                               # 总资产
                "total_liabilities",                          # 总负债
                "shareholders_equity",                        # 股东权益
                "dividends_and_other_cash_distributions",     # 股息和其他现金分配
                "issuance_or_purchase_of_equity_shares",      # 股权发行或回购
                "gross_profit",                               # 毛利润
                "revenue",                                    # 收入
                "free_cash_flow",                             # 自由现金流
            ],
            end_date,
            period="ttm",  # 过去十二个月数据
            limit=10,       # 限制10个期间
        )

        # 更新进度状态：获取市值
        progress.update_status("warren_buffett_agent", ticker, "获取市值")
        # 获取当前市值
        market_cap = get_market_cap(ticker, end_date)

        # 更新进度状态：分析基本面
        progress.update_status("warren_buffett_agent", ticker, "分析基本面")
        # 分析基本面指标
        fundamental_analysis = analyze_fundamentals(metrics)

        # 更新进度状态：分析一致性
        progress.update_status("warren_buffett_agent", ticker, "分析一致性")
        # 分析财务表现的一致性
        consistency_analysis = analyze_consistency(financial_line_items)

        # 更新进度状态：分析竞争护城河
        progress.update_status("warren_buffett_agent", ticker, "分析竞争护城河")
        # 分析公司的竞争优势
        moat_analysis = analyze_moat(metrics)

        # 更新进度状态：分析定价能力
        progress.update_status("warren_buffett_agent", ticker, "分析定价能力")
        # 分析公司的定价能力
        pricing_power_analysis = analyze_pricing_power(financial_line_items, metrics)

        # 更新进度状态：分析账面价值增长
        progress.update_status("warren_buffett_agent", ticker, "分析账面价值增长")
        # 分析账面价值的增长情况
        book_value_analysis = analyze_book_value_growth(financial_line_items)

        # 更新进度状态：分析管理层质量
        progress.update_status("warren_buffett_agent", ticker, "分析管理层质量")
        # 分析管理层的质量
        mgmt_analysis = analyze_management_quality(financial_line_items)

        # 更新进度状态：计算内在价值
        progress.update_status("warren_buffett_agent", ticker, "计算内在价值")
        # 计算股票的内在价值
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # 计算总分（不包括能力圈，由大语言模型处理）
        total_score = (
            fundamental_analysis["score"] +     # 基本面分析得分
            consistency_analysis["score"] +    # 一致性分析得分
            moat_analysis["score"] +           # 护城河分析得分
            mgmt_analysis["score"] +           # 管理层质量得分
            pricing_power_analysis["score"] +  # 定价能力得分
            book_value_analysis["score"]       # 账面价值增长得分
        )
        
        # 更新最大可能得分计算
        max_possible_score = (
            10 +  # 基本面分析（ROE、债务、利润率、流动比率）
            moat_analysis["max_score"] +       # 护城河最大得分
            mgmt_analysis["max_score"] +       # 管理层质量最大得分
            5 +   # 定价能力（0-5分）
            5     # 账面价值增长（0-5分）
        )

        # 如果同时有内在价值和当前价格，则添加安全边际分析
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            # 计算安全边际：(内在价值 - 市值) / 市值
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

        # 合并所有分析结果用于大语言模型评估
        analysis_data[ticker] = {
            "ticker": ticker,                                    # 股票代码
            "score": total_score,                               # 总得分
            "max_score": max_possible_score,                    # 最大可能得分
            "fundamental_analysis": fundamental_analysis,       # 基本面分析
            "consistency_analysis": consistency_analysis,       # 一致性分析
            "moat_analysis": moat_analysis,                     # 护城河分析
            "pricing_power_analysis": pricing_power_analysis,   # 定价能力分析
            "book_value_analysis": book_value_analysis,         # 账面价值分析
            "management_analysis": mgmt_analysis,               # 管理层分析
            "intrinsic_value_analysis": intrinsic_value_analysis, # 内在价值分析
            "market_cap": market_cap,                           # 市值
            "margin_of_safety": margin_of_safety,               # 安全边际
        }

        # 更新进度状态：生成巴菲特分析
        progress.update_status("warren_buffett_agent", ticker, "生成巴菲特分析")
        # 生成巴菲特风格的分析输出
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            state=state,
        )

        # 以与其他代理一致的格式存储分析结果
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,        # 信号类型
            "confidence": buffett_output.confidence, # 置信度
            "reasoning": buffett_output.reasoning,   # 推理过程
        }

        # 更新进度状态为完成
        progress.update_status("warren_buffett_agent", ticker, "完成", analysis=buffett_output.reasoning)

    # 创建消息
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    # 如果请求显示推理过程，则显示
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "巴菲特代理")

    # 将信号添加到分析师信号列表中
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    # 更新进度状态为完成
    progress.update_status("warren_buffett_agent", None, "完成")

    # 返回更新后的状态
    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """基于巴菲特标准分析公司基本面
    
    Args:
        metrics: 财务指标列表
        
    Returns:
        dict: 包含得分和详细信息的分析结果
    """
    if not metrics:
        return {"score": 0, "details": "基本面数据不足"}

    latest_metrics = metrics[0]  # 获取最新的财务指标

    score = 0      # 初始化得分
    reasoning = [] # 初始化推理列表

    # 检查ROE（股本回报率）
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:  # 15%的ROE阈值
        score += 2
        reasoning.append(f"强劲的ROE为{latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"较弱的ROE为{latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE数据不可用")

    # 检查债务股权比
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("保守的债务水平")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"较高的债务股权比为{latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("债务股权比数据不可用")

    # 检查营业利润率
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # Check earnings growth trend
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        # Simple check: is each period's earnings bigger than the next?
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate total growth rate from oldest to latest
        if len(earnings_values) >= 2 and earnings_values[-1] != 0:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def analyze_moat(metrics: list) -> dict[str, any]:
    """
    Evaluate whether the company likely has a durable competitive advantage (moat).
    Enhanced to include multiple moat indicators that Buffett actually looks for:
    1. Consistent high returns on capital
    2. Pricing power (stable/growing margins)
    3. Scale advantages (improving metrics with size)
    4. Brand strength (inferred from margins and consistency)
    5. Switching costs (inferred from customer retention)
    """
    if not metrics or len(metrics) < 5:  # Need more data for proper moat analysis
        return {"score": 0, "max_score": 5, "details": "Insufficient data for comprehensive moat analysis"}

    reasoning = []
    moat_score = 0
    max_score = 5

    # 1. Return on Capital Consistency (Buffett's favorite moat indicator)
    historical_roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    historical_roics = [m.return_on_invested_capital for m in metrics if hasattr(m, 'return_on_invested_capital') and m.return_on_invested_capital is not None]
    
    if len(historical_roes) >= 5:
        # Check for consistently high ROE (>15% for most periods)
        high_roe_periods = sum(1 for roe in historical_roes if roe > 0.15)
        roe_consistency = high_roe_periods / len(historical_roes)
        
        if roe_consistency >= 0.8:  # 80%+ of periods with ROE > 15%
            moat_score += 2
            avg_roe = sum(historical_roes) / len(historical_roes)
            reasoning.append(f"Excellent ROE consistency: {high_roe_periods}/{len(historical_roes)} periods >15% (avg: {avg_roe:.1%}) - indicates durable competitive advantage")
        elif roe_consistency >= 0.6:
            moat_score += 1
            reasoning.append(f"Good ROE performance: {high_roe_periods}/{len(historical_roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high_roe_periods}/{len(historical_roes)} periods >15%")
    else:
        reasoning.append("Insufficient ROE history for moat analysis")

    # 2. Operating Margin Stability (Pricing Power Indicator)
    historical_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(historical_margins) >= 5:
        # Check for stable or improving margins (sign of pricing power)
        avg_margin = sum(historical_margins) / len(historical_margins)
        recent_margins = historical_margins[:3]  # Last 3 periods
        older_margins = historical_margins[-3:]  # First 3 periods
        
        recent_avg = sum(recent_margins) / len(recent_margins)
        older_avg = sum(older_margins) / len(older_margins)
        
        if avg_margin > 0.2 and recent_avg >= older_avg:  # 20%+ margins and stable/improving
            moat_score += 1
            reasoning.append(f"Strong and stable operating margins (avg: {avg_margin:.1%}) indicate pricing power moat")
        elif avg_margin > 0.15:  # At least decent margins
            reasoning.append(f"Decent operating margins (avg: {avg_margin:.1%}) suggest some competitive advantage")
        else:
            reasoning.append(f"Low operating margins (avg: {avg_margin:.1%}) suggest limited pricing power")
    
    # 3. Asset Efficiency and Scale Advantages
    if len(metrics) >= 5:
        # Check asset turnover trends (revenue efficiency)
        asset_turnovers = []
        for m in metrics:
            if hasattr(m, 'asset_turnover') and m.asset_turnover is not None:
                asset_turnovers.append(m.asset_turnover)
        
        if len(asset_turnovers) >= 3:
            if any(turnover > 1.0 for turnover in asset_turnovers):  # Efficient asset use
                moat_score += 1
                reasoning.append("Efficient asset utilization suggests operational moat")
    
    # 4. Competitive Position Strength (inferred from trend stability)
    if len(historical_roes) >= 5 and len(historical_margins) >= 5:
        # Calculate coefficient of variation (stability measure)
        roe_avg = sum(historical_roes) / len(historical_roes)
        roe_variance = sum((roe - roe_avg) ** 2 for roe in historical_roes) / len(historical_roes)
        roe_stability = 1 - (roe_variance ** 0.5) / roe_avg if roe_avg > 0 else 0
        
        margin_avg = sum(historical_margins) / len(historical_margins)
        margin_variance = sum((margin - margin_avg) ** 2 for margin in historical_margins) / len(historical_margins)
        margin_stability = 1 - (margin_variance ** 0.5) / margin_avg if margin_avg > 0 else 0
        
        overall_stability = (roe_stability + margin_stability) / 2
        
        if overall_stability > 0.7:  # High stability indicates strong competitive position
            moat_score += 1
            reasoning.append(f"High performance stability ({overall_stability:.1%}) suggests strong competitive moat")
    
    # Cap the score at max_score
    moat_score = min(moat_score, max_score)

    return {
        "score": moat_score,
        "max_score": max_score,
        "details": "; ".join(reasoning) if reasoning else "Limited moat analysis available",
    }


def analyze_management_quality(financial_line_items: list) -> dict[str, any]:
    """
    Checks for share dilution or consistent buybacks, and some dividend track record.
    A simplified approach:
      - if there's net share repurchase or stable share count, it suggests management
        might be shareholder-friendly.
      - if there's a big new issuance, it might be a negative sign (dilution).
    """
    if not financial_line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

    reasoning = []
    mgmt_score = 0

    latest = financial_line_items[0]
    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares < 0:
        # Negative means the company spent money on buybacks
        mgmt_score += 1
        reasoning.append("Company has been repurchasing shares (shareholder-friendly)")

    if hasattr(latest, "issuance_or_purchase_of_equity_shares") and latest.issuance_or_purchase_of_equity_shares and latest.issuance_or_purchase_of_equity_shares > 0:
        # Positive issuance means new shares => possible dilution
        reasoning.append("Recent common stock issuance (potential dilution)")
    else:
        reasoning.append("No significant new stock issuance detected")

    # Check for any dividends
    if hasattr(latest, "dividends_and_other_cash_distributions") and latest.dividends_and_other_cash_distributions and latest.dividends_and_other_cash_distributions < 0:
        mgmt_score += 1
        reasoning.append("Company has a track record of paying dividends")
    else:
        reasoning.append("No or minimal dividends paid")

    return {
        "score": mgmt_score,
        "max_score": 2,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """
    Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Enhanced methodology: Net Income + Depreciation/Amortization - Maintenance CapEx - Working Capital Changes
    Uses multi-period analysis for better maintenance capex estimation.
    """
    if not financial_line_items or len(financial_line_items) < 2:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]
    details = []

    # Core components
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income is not None, depreciation is not None, capex is not None]):
        missing = []
        if net_income is None: missing.append("net income")
        if depreciation is None: missing.append("depreciation")
        if capex is None: missing.append("capital expenditure")
        return {"owner_earnings": None, "details": [f"Missing components: {', '.join(missing)}"]}

    # Enhanced maintenance capex estimation using historical analysis
    maintenance_capex = estimate_maintenance_capex(financial_line_items)
    
    # Working capital change analysis (if data available)
    working_capital_change = 0
    if len(financial_line_items) >= 2:
        try:
            current_assets_current = getattr(latest, 'current_assets', None)
            current_liab_current = getattr(latest, 'current_liabilities', None)
            
            previous = financial_line_items[1]
            current_assets_previous = getattr(previous, 'current_assets', None)
            current_liab_previous = getattr(previous, 'current_liabilities', None)
            
            if all([current_assets_current, current_liab_current, current_assets_previous, current_liab_previous]):
                wc_current = current_assets_current - current_liab_current
                wc_previous = current_assets_previous - current_liab_previous
                working_capital_change = wc_current - wc_previous
                details.append(f"Working capital change: ${working_capital_change:,.0f}")
        except:
            pass  # Skip working capital adjustment if data unavailable

    # Calculate owner earnings
    owner_earnings = net_income + depreciation - maintenance_capex - working_capital_change

    # Sanity checks
    if owner_earnings < net_income * 0.3:  # Owner earnings shouldn't be less than 30% of net income typically
        details.append("Warning: Owner earnings significantly below net income - high capex intensity")
    
    if maintenance_capex > depreciation * 2:  # Maintenance capex shouldn't typically exceed 2x depreciation
        details.append("Warning: Estimated maintenance capex seems high relative to depreciation")

    details.extend([
        f"Net income: ${net_income:,.0f}",
        f"Depreciation: ${depreciation:,.0f}",
        f"Estimated maintenance capex: ${maintenance_capex:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}"
    ])

    return {
        "owner_earnings": owner_earnings,
        "components": {
            "net_income": net_income,
            "depreciation": depreciation,
            "maintenance_capex": maintenance_capex,
            "working_capital_change": working_capital_change,
            "total_capex": abs(capex) if capex else 0
        },
        "details": details,
    }


def estimate_maintenance_capex(financial_line_items: list) -> float:
    """
    Estimate maintenance capital expenditure using multiple approaches.
    Buffett considers this crucial for understanding true owner earnings.
    """
    if not financial_line_items:
        return 0
    
    # Approach 1: Historical average as % of revenue
    capex_ratios = []
    depreciation_values = []
    
    for item in financial_line_items[:5]:  # Last 5 periods
        if hasattr(item, 'capital_expenditure') and hasattr(item, 'revenue'):
            if item.capital_expenditure and item.revenue and item.revenue > 0:
                capex_ratio = abs(item.capital_expenditure) / item.revenue
                capex_ratios.append(capex_ratio)
        
        if hasattr(item, 'depreciation_and_amortization') and item.depreciation_and_amortization:
            depreciation_values.append(item.depreciation_and_amortization)
    
    # Approach 2: Percentage of depreciation (typically 80-120% for maintenance)
    latest_depreciation = financial_line_items[0].depreciation_and_amortization if financial_line_items[0].depreciation_and_amortization else 0
    
    # Approach 3: Industry-specific heuristics
    latest_capex = abs(financial_line_items[0].capital_expenditure) if financial_line_items[0].capital_expenditure else 0
    
    # Conservative estimate: Use the higher of:
    # 1. 85% of total capex (assuming 15% is growth capex)
    # 2. 100% of depreciation (replacement of worn-out assets)
    # 3. Historical average if stable
    
    method_1 = latest_capex * 0.85  # 85% of total capex
    method_2 = latest_depreciation  # 100% of depreciation
    
    # If we have historical data, use average capex ratio
    if len(capex_ratios) >= 3:
        avg_capex_ratio = sum(capex_ratios) / len(capex_ratios)
        latest_revenue = financial_line_items[0].revenue if hasattr(financial_line_items[0], 'revenue') and financial_line_items[0].revenue else 0
        method_3 = avg_capex_ratio * latest_revenue if latest_revenue else 0
        
        # Use the median of the three approaches for conservatism
        estimates = sorted([method_1, method_2, method_3])
        return estimates[1]  # Median
    else:
        # Use the higher of method 1 and 2
        return max(method_1, method_2)


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """
    Calculate intrinsic value using enhanced DCF with owner earnings.
    Uses more sophisticated assumptions and conservative approach like Buffett.
    """
    if not financial_line_items or len(financial_line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data for reliable valuation"]}

    # Calculate owner earnings with better methodology
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"intrinsic_value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding or shares_outstanding <= 0:
        return {"intrinsic_value": None, "details": ["Missing or invalid shares outstanding data"]}

    # Enhanced DCF with more realistic assumptions
    details = []
    
    # Estimate growth rate based on historical performance (more conservative)
    historical_earnings = []
    for item in financial_line_items[:5]:  # Last 5 years
        if hasattr(item, 'net_income') and item.net_income:
            historical_earnings.append(item.net_income)
    
    # Calculate historical growth rate
    if len(historical_earnings) >= 3:
        oldest_earnings = historical_earnings[-1]
        latest_earnings = historical_earnings[0]
        years = len(historical_earnings) - 1
        
        if oldest_earnings > 0:
            historical_growth = ((latest_earnings / oldest_earnings) ** (1/years)) - 1
            # Conservative adjustment - cap growth and apply haircut
            historical_growth = max(-0.05, min(historical_growth, 0.15))  # Cap between -5% and 15%
            conservative_growth = historical_growth * 0.7  # Apply 30% haircut for conservatism
        else:
            conservative_growth = 0.03  # Default 3% if negative base
    else:
        conservative_growth = 0.03  # Default conservative growth
    
    # Buffett's conservative assumptions
    stage1_growth = min(conservative_growth, 0.08)  # Stage 1: cap at 8%
    stage2_growth = min(conservative_growth * 0.5, 0.04)  # Stage 2: half of stage 1, cap at 4%
    terminal_growth = 0.025  # Long-term GDP growth rate
    
    # Risk-adjusted discount rate based on business quality
    base_discount_rate = 0.09  # Base 9%
    
    # Adjust based on analysis scores (if available in calling context)
    # For now, use conservative 10%
    discount_rate = 0.10
    
    # Three-stage DCF model
    stage1_years = 5   # High growth phase
    stage2_years = 5   # Transition phase
    
    present_value = 0
    details.append(f"Using three-stage DCF: Stage 1 ({stage1_growth:.1%}, {stage1_years}y), Stage 2 ({stage2_growth:.1%}, {stage2_years}y), Terminal ({terminal_growth:.1%})")
    
    # Stage 1: Higher growth
    stage1_pv = 0
    for year in range(1, stage1_years + 1):
        future_earnings = owner_earnings * (1 + stage1_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** year
        stage1_pv += pv
    
    # Stage 2: Transition growth
    stage2_pv = 0
    stage1_final_earnings = owner_earnings * (1 + stage1_growth) ** stage1_years
    for year in range(1, stage2_years + 1):
        future_earnings = stage1_final_earnings * (1 + stage2_growth) ** year
        pv = future_earnings / (1 + discount_rate) ** (stage1_years + year)
        stage2_pv += pv
    
    # Terminal value using Gordon Growth Model
    final_earnings = stage1_final_earnings * (1 + stage2_growth) ** stage2_years
    terminal_earnings = final_earnings * (1 + terminal_growth)
    terminal_value = terminal_earnings / (discount_rate - terminal_growth)
    terminal_pv = terminal_value / (1 + discount_rate) ** (stage1_years + stage2_years)
    
    # Total intrinsic value
    intrinsic_value = stage1_pv + stage2_pv + terminal_pv
    
    # Apply additional margin of safety (Buffett's conservatism)
    conservative_intrinsic_value = intrinsic_value * 0.85  # 15% additional haircut
    
    details.extend([
        f"Stage 1 PV: ${stage1_pv:,.0f}",
        f"Stage 2 PV: ${stage2_pv:,.0f}",
        f"Terminal PV: ${terminal_pv:,.0f}",
        f"Total IV: ${intrinsic_value:,.0f}",
        f"Conservative IV (15% haircut): ${conservative_intrinsic_value:,.0f}",
        f"Owner earnings: ${owner_earnings:,.0f}",
        f"Discount rate: {discount_rate:.1%}"
    ])

    return {
        "intrinsic_value": conservative_intrinsic_value,
        "raw_intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "stage1_growth": stage1_growth,
            "stage2_growth": stage2_growth,
            "terminal_growth": terminal_growth,
            "discount_rate": discount_rate,
            "stage1_years": stage1_years,
            "stage2_years": stage2_years,
            "historical_growth": conservative_growth if 'conservative_growth' in locals() else None,
        },
        "details": details,
    }

def analyze_book_value_growth(financial_line_items: list) -> dict[str, any]:
    """
    Analyze book value per share growth - a key Buffett metric for long-term value creation.
    Buffett often talks about companies that compound book value over decades.
    """
    if len(financial_line_items) < 3:
        return {"score": 0, "details": "Insufficient data for book value analysis"}
    
    score = 0
    reasoning = []
    
    # Calculate book value growth (shareholders equity / shares outstanding)
    book_values = []
    for item in financial_line_items:
        if hasattr(item, 'shareholders_equity') and hasattr(item, 'outstanding_shares'):
            if item.shareholders_equity and item.outstanding_shares:
                book_value_per_share = item.shareholders_equity / item.outstanding_shares
                book_values.append(book_value_per_share)
    
    if len(book_values) >= 3:
        # Check for consistent book value growth
        growth_periods = 0
        for i in range(len(book_values) - 1):
            if book_values[i] > book_values[i + 1]:  # Current > Previous (reverse chronological)
                growth_periods += 1
        
        growth_rate = growth_periods / (len(book_values) - 1)
        
        if growth_rate >= 0.8:  # 80% of periods show growth
            score += 3
            reasoning.append("Consistent book value per share growth (Buffett's favorite metric)")
        elif growth_rate >= 0.6:
            score += 2
            reasoning.append("Good book value per share growth pattern")
        elif growth_rate >= 0.4:
            score += 1
            reasoning.append("Moderate book value per share growth")
        else:
            reasoning.append("Inconsistent book value per share growth")
            
        # Calculate compound annual growth rate
        if len(book_values) >= 2:
            oldest_bv = book_values[-1]
            latest_bv = book_values[0]
            years = len(book_values) - 1
            if oldest_bv > 0:
                cagr = ((latest_bv / oldest_bv) ** (1/years)) - 1
                if cagr > 0.15:  # 15%+ CAGR
                    score += 2
                    reasoning.append(f"Excellent book value CAGR: {cagr:.1%}")
                elif cagr > 0.1:  # 10%+ CAGR
                    score += 1
                    reasoning.append(f"Good book value CAGR: {cagr:.1%}")
    else:
        reasoning.append("Insufficient book value data for growth analysis")
    
    return {
        "score": score,
        "details": "; ".join(reasoning)
    }


def analyze_pricing_power(financial_line_items: list, metrics: list) -> dict[str, any]:
    """
    Analyze pricing power - Buffett's key indicator of a business moat.
    Looks at ability to raise prices without losing customers (margin expansion during inflation).
    """
    if not financial_line_items or not metrics:
        return {"score": 0, "details": "Insufficient data for pricing power analysis"}
    
    score = 0
    reasoning = []
    
    # Check gross margin trends (ability to maintain/expand margins)
    gross_margins = []
    for item in financial_line_items:
        if hasattr(item, 'gross_margin') and item.gross_margin is not None:
            gross_margins.append(item.gross_margin)
    
    if len(gross_margins) >= 3:
        # Check margin stability/improvement
        recent_avg = sum(gross_margins[:2]) / 2 if len(gross_margins) >= 2 else gross_margins[0]
        older_avg = sum(gross_margins[-2:]) / 2 if len(gross_margins) >= 2 else gross_margins[-1]
        
        if recent_avg > older_avg + 0.02:  # 2%+ improvement
            score += 3
            reasoning.append("Expanding gross margins indicate strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving gross margins suggest good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:  # Stable within 1%
            score += 1
            reasoning.append("Stable gross margins during economic uncertainty")
        else:
            reasoning.append("Declining gross margins may indicate pricing pressure")
    
    # Check if company has been able to maintain high margins consistently
    if gross_margins:
        avg_margin = sum(gross_margins) / len(gross_margins)
        if avg_margin > 0.5:  # 50%+ gross margins
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg_margin:.1%}) indicate strong pricing power")
        elif avg_margin > 0.3:  # 30%+ gross margins
            score += 1
            reasoning.append(f"Good gross margins ({avg_margin:.1%}) suggest decent pricing power")
    
    return {
        "score": score,
        "details": "; ".join(reasoning) if reasoning else "Limited pricing power analysis available"
    }


def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    state: AgentState,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles"""
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Warren Buffett, the Oracle of Omaha. Analyze investment opportunities using my proven methodology developed over 60+ years of investing:

                MY CORE PRINCIPLES:
                1. Circle of Competence: "Risk comes from not knowing what you're doing." Only invest in businesses I thoroughly understand.
                2. Economic Moats: Seek companies with durable competitive advantages - pricing power, brand strength, scale advantages, switching costs.
                3. Quality Management: Look for honest, competent managers who think like owners and allocate capital wisely.
                4. Financial Fortress: Prefer companies with strong balance sheets, consistent earnings, and minimal debt.
                5. Intrinsic Value & Margin of Safety: Pay significantly less than what the business is worth - "Price is what you pay, value is what you get."
                6. Long-term Perspective: "Our favorite holding period is forever." Look for businesses that will prosper for decades.
                7. Pricing Power: The best businesses can raise prices without losing customers.

                MY CIRCLE OF COMPETENCE PREFERENCES:
                STRONGLY PREFER:
                - Consumer staples with strong brands (Coca-Cola, P&G, Walmart, Costco)
                - Commercial banking (Bank of America, Wells Fargo) - NOT investment banking
                - Insurance (GEICO, property & casualty)
                - Railways and utilities (BNSF, simple infrastructure)
                - Simple industrials with moats (UPS, FedEx, Caterpillar)
                - Energy companies with reserves and pipelines (Chevron, not exploration)

                GENERALLY AVOID:
                - Complex technology (semiconductors, software, except Apple due to consumer ecosystem)
                - Biotechnology and pharmaceuticals (too complex, regulatory risk)
                - Airlines (commodity business, poor economics)
                - Cryptocurrency and fintech speculation
                - Complex derivatives or financial instruments
                - Rapid technology change industries
                - Capital-intensive businesses without pricing power

                APPLE EXCEPTION: I own Apple not as a tech stock, but as a consumer products company with an ecosystem that creates switching costs.

                MY INVESTMENT CRITERIA HIERARCHY:
                First: Circle of Competence - If I don't understand the business model or industry dynamics, I don't invest, regardless of potential returns.
                Second: Business Quality - Does it have a moat? Will it still be thriving in 20 years?
                Third: Management - Do they act in shareholders' interests? Smart capital allocation?
                Fourth: Financial Strength - Consistent earnings, low debt, strong returns on capital?
                Fifth: Valuation - Am I paying a reasonable price for this wonderful business?

                MY LANGUAGE & STYLE:
                - Use folksy wisdom and simple analogies ("It's like...")
                - Reference specific past investments when relevant (Coca-Cola, Apple, GEICO, See's Candies, etc.)
                - Quote my own sayings when appropriate
                - Be candid about what I don't understand
                - Show patience - most opportunities don't meet my criteria
                - Express genuine enthusiasm for truly exceptional businesses
                - Be skeptical of complexity and Wall Street jargon

                CONFIDENCE LEVELS:
                - 90-100%: Exceptional business within my circle, trading at attractive price
                - 70-89%: Good business with decent moat, fair valuation
                - 50-69%: Mixed signals, would need more information or better price
                - 30-49%: Outside my expertise or concerning fundamentals
                - 10-29%: Poor business or significantly overvalued

                Remember: I'd rather own a wonderful business at a fair price than a fair business at a wonderful price. And when in doubt, the answer is usually "no" - there's no penalty for missed opportunities, only for permanent capital loss.
                """,
            ),
            (
                "human",
                """Analyze this investment opportunity for {ticker}:

                COMPREHENSIVE ANALYSIS DATA:
                {analysis_data}

                Please provide your investment decision in exactly this JSON format:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string with your detailed Warren Buffett-style analysis"
                }}

                In your reasoning, be specific about:
                1. Whether this falls within your circle of competence and why (CRITICAL FIRST STEP)
                2. Your assessment of the business's competitive moat
                3. Management quality and capital allocation
                4. Financial health and consistency
                5. Valuation relative to intrinsic value
                6. Long-term prospects and any red flags
                7. How this compares to opportunities in your portfolio

                Write as Warren Buffett would speak - plainly, with conviction, and with specific references to the data provided.
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt,
        pydantic_model=WarrenBuffettSignal,
        agent_name="warren_buffett_agent",
        state=state,
        default_factory=create_default_warren_buffett_signal,
    )
