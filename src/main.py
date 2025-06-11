# 导入系统模块
import sys

# 导入环境变量加载模块
from dotenv import load_dotenv
# 导入LangChain核心消息类
from langchain_core.messages import HumanMessage
# 导入LangGraph图形构建模块
from langgraph.graph import END, StateGraph
# 导入颜色输出模块
from colorama import Fore, Style, init
# 导入交互式问答模块
import questionary
# 导入投资组合管理代理
from src.agents.portfolio_manager import portfolio_management_agent
# 导入风险管理代理
from src.agents.risk_manager import risk_management_agent
# 导入代理状态管理
from src.graph.state import AgentState
# 导入交易输出显示工具
from src.utils.display import print_trading_output
# 导入分析师相关工具
from src.utils.analysts import ANALYST_ORDER, get_analyst_nodes
# 导入进度显示工具
from src.utils.progress import progress
# 导入大语言模型相关配置
from src.llm.models import LLM_ORDER, OLLAMA_LLM_ORDER, get_model_info, ModelProvider
# 导入Ollama模型管理工具
from src.utils.ollama import ensure_ollama_and_model

# 导入命令行参数解析模块
import argparse
# 导入日期时间处理模块
from datetime import datetime
from dateutil.relativedelta import relativedelta
# 导入图形可视化工具
from src.utils.visualize import save_graph_as_png
# 导入JSON处理模块
import json

# 从.env文件加载环境变量
load_dotenv()

# 初始化colorama，用于彩色终端输出
init(autoreset=True)


def parse_hedge_fund_response(response):
    """解析对冲基金响应的JSON字符串并返回字典
    
    Args:
        response: 需要解析的响应字符串
        
    Returns:
        dict: 解析后的字典，如果解析失败则返回None
    """
    try:
        # 尝试将JSON字符串解析为Python字典
        return json.loads(response)
    except json.JSONDecodeError as e:
        # 处理JSON解码错误
        print(f"JSON解码错误: {e}\n响应内容: {repr(response)}")
        return None
    except TypeError as e:
        # 处理类型错误（期望字符串但得到其他类型）
        print(f"无效的响应类型 (期望字符串，得到 {type(response).__name__}): {e}")
        return None
    except Exception as e:
        # 处理其他未预期的错误
        print(f"解析响应时发生未预期错误: {e}\n响应内容: {repr(response)}")
        return None


##### 运行对冲基金系统 #####
def run_hedge_fund(
    tickers: list[str],          # 股票代码列表
    start_date: str,             # 开始日期
    end_date: str,               # 结束日期
    portfolio: dict,             # 投资组合字典
    show_reasoning: bool = False, # 是否显示推理过程
    selected_analysts: list[str] = [], # 选择的分析师列表
    model_name: str = "gpt-4o",  # 使用的模型名称
    model_provider: str = "OpenAI", # 模型提供商
):
    """运行AI对冲基金系统的主函数
    
    Args:
        tickers: 要分析的股票代码列表
        start_date: 分析开始日期
        end_date: 分析结束日期
        portfolio: 当前投资组合
        show_reasoning: 是否显示各代理的推理过程
        selected_analysts: 选择使用的分析师代理列表
        model_name: 使用的大语言模型名称
        model_provider: 模型提供商名称
    """
    # 开始进度跟踪
    progress.start()

    try:
        # 如果自定义了分析师，则创建新的工作流
        if selected_analysts:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            # 使用默认的应用工作流
            agent = app

        # 调用代理执行交易决策分析
        final_state = agent.invoke(
            {
                # 消息列表，包含初始指令
                "messages": [
                    HumanMessage(
                        content="基于提供的数据做出交易决策。",
                    )
                ],
                # 数据字典，包含分析所需的所有信息
                "data": {
                    "tickers": tickers,           # 股票代码列表
                    "portfolio": portfolio,       # 当前投资组合
                    "start_date": start_date,     # 开始日期
                    "end_date": end_date,         # 结束日期
                    "analyst_signals": {},        # 分析师信号（初始为空）
                },
                # 元数据，包含配置信息
                "metadata": {
                    "show_reasoning": show_reasoning,   # 是否显示推理过程
                    "model_name": model_name,           # 模型名称
                    "model_provider": model_provider,   # 模型提供商
                },
            },
        )

        # 返回最终的交易决策和分析师信号
        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # 停止进度跟踪
        progress.stop()


def start(state: AgentState):
    """初始化工作流的起始状态
    
    Args:
        state: 代理状态对象
        
    Returns:
        AgentState: 返回未修改的状态对象
    """
    return state


def create_workflow(selected_analysts=None):
    """创建包含选定分析师的工作流
    
    Args:
        selected_analysts: 选择的分析师列表，如果为None则使用所有分析师
        
    Returns:
        StateGraph: 配置好的状态图工作流
    """
    # 创建状态图工作流
    workflow = StateGraph(AgentState)
    # 添加起始节点
    workflow.add_node("start_node", start)

    # 从配置中获取分析师节点
    analyst_nodes = get_analyst_nodes()

    # 如果没有选择分析师，则默认使用所有分析师
    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())
    # 添加选定的分析师节点
    for analyst_key in selected_analysts:
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)  # 添加分析师节点
        workflow.add_edge("start_node", node_name)  # 连接起始节点到分析师节点

    # 始终添加风险管理和投资组合管理节点
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    # 将选定的分析师连接到风险管理节点
    for analyst_key in selected_analysts:
        node_name = analyst_nodes[analyst_key][0]
        workflow.add_edge(node_name, "risk_management_agent")

    # 连接风险管理到投资组合管理
    workflow.add_edge("risk_management_agent", "portfolio_manager")
    # 连接投资组合管理到结束节点
    workflow.add_edge("portfolio_manager", END)

    # 设置工作流的入口点
    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="运行对冲基金交易系统")
    # 初始现金位置参数
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="初始现金位置，默认为100000.0")
    # 保证金要求参数
    parser.add_argument("--margin-requirement", type=float, default=0.0, help="初始保证金要求，默认为0.0")
    # 股票代码参数（必需）
    parser.add_argument("--tickers", type=str, required=True, help="逗号分隔的股票代码列表")
    # 开始日期参数
    parser.add_argument(
        "--start-date",
        type=str,
        help="开始日期 (YYYY-MM-DD)，默认为结束日期前3个月",
    )
    # 结束日期参数
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)，默认为今天")
    # 显示推理过程参数
    parser.add_argument("--show-reasoning", action="store_true", help="显示每个代理的推理过程")
    # 显示代理图参数
    parser.add_argument("--show-agent-graph", action="store_true", help="显示代理关系图")
    # 使用Ollama本地推理参数
    parser.add_argument("--ollama", action="store_true", help="使用Ollama进行本地大语言模型推理")

    # 解析命令行参数
    args = parser.parse_args()

    # 从逗号分隔的字符串中解析股票代码
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # 选择分析师
    selected_analysts = None
    choices = questionary.checkbox(
        "选择您的AI分析师。",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        instruction="\n\n操作说明: \n1. 按空格键选择/取消选择分析师。\n2. 按'a'键全选/全不选。\n3. 完成选择后按回车键运行对冲基金。\n",
        validate=lambda x: len(x) > 0 or "您必须至少选择一个分析师。",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),  # 已选择的复选框样式
                ("selected", "fg:green noinherit"),  # 选中项样式
                ("highlighted", "noinherit"),        # 高亮样式
                ("pointer", "noinherit"),            # 指针样式
            ]
        ),
    ).ask()

    # 检查用户是否取消了选择
    if not choices:
        print("\n\n收到中断信号，正在退出...")
        sys.exit(0)
    else:
        selected_analysts = choices
        # 显示选择的分析师
        print(f"\n选择的分析师: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n")

    # 根据是否使用Ollama选择大语言模型
    model_name = ""
    model_provider = ""

    if args.ollama:
        print(f"{Fore.CYAN}使用Ollama进行本地大语言模型推理。{Style.RESET_ALL}")

        # 从Ollama专用模型中选择
        model_name: str = questionary.select(
            "选择您的Ollama模型:",
            choices=[questionary.Choice(display, value=value) for display, value, _ in OLLAMA_LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),    # 选中项样式
                    ("pointer", "fg:green bold"),     # 指针样式
                    ("highlighted", "fg:green"),      # 高亮样式
                    ("answer", "fg:green bold"),      # 答案样式
                ]
            ),
        ).ask()

        # 检查用户是否取消了选择
        if not model_name:
            print("\n\n收到中断信号，正在退出...")
            sys.exit(0)

        # 如果选择了自定义模型选项
        if model_name == "-":
            model_name = questionary.text("输入自定义模型名称:").ask()
            if not model_name:
                print("\n\n收到中断信号，正在退出...")
                sys.exit(0)

        # 确保Ollama已安装、运行且模型可用
        if not ensure_ollama_and_model(model_name):
            print(f"{Fore.RED}无法在没有Ollama和选定模型的情况下继续。{Style.RESET_ALL}")
            sys.exit(1)

        # 设置模型提供商为Ollama
        model_provider = ModelProvider.OLLAMA.value
        print(f"\n选择的 {Fore.CYAN}Ollama{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
    else:
        # 使用标准的云端大语言模型选择
        model_choice = questionary.select(
            "选择您的大语言模型:",
            choices=[questionary.Choice(display, value=(name, provider)) for display, name, provider in LLM_ORDER],
            style=questionary.Style(
                [
                    ("selected", "fg:green bold"),    # 选中项样式
                    ("pointer", "fg:green bold"),     # 指针样式
                    ("highlighted", "fg:green"),      # 高亮样式
                    ("answer", "fg:green bold"),      # 答案样式
                ]
            ),
        ).ask()

        # 检查用户是否取消了选择
        if not model_choice:
            print("\n\n收到中断信号，正在退出...")
            sys.exit(0)

        # 解包模型名称和提供商
        model_name, model_provider = model_choice

        # 使用辅助函数获取模型信息
        model_info = get_model_info(model_name, model_provider)
        if model_info:
            # 如果是自定义模型，需要用户输入模型名称
            if model_info.is_custom():
                model_name = questionary.text("输入自定义模型名称:").ask()
                if not model_name:
                    print("\n\n收到中断信号，正在退出...")
                    sys.exit(0)

            # 显示选择的模型信息
            print(f"\n选择的 {Fore.CYAN}{model_provider}{Style.RESET_ALL} 模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")
        else:
            # 如果无法获取模型信息，设置为未知提供商
            model_provider = "Unknown"
            print(f"\n选择的模型: {Fore.GREEN + Style.BRIGHT}{model_name}{Style.RESET_ALL}\n")

    # 使用选定的分析师创建工作流
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    # 如果需要显示代理图，则保存为PNG文件
    if args.show_agent_graph:
        file_path = ""
        if selected_analysts is not None:
            # 根据选择的分析师生成文件名
            for selected_analyst in selected_analysts:
                file_path += selected_analyst + "_"
            file_path += "graph.png"
        # 保存代理关系图
        save_graph_as_png(app, file_path)

    # 验证提供的日期格式
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("开始日期必须是YYYY-MM-DD格式")

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("结束日期必须是YYYY-MM-DD格式")

    # 设置开始和结束日期
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # 计算结束日期前3个月作为开始日期
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # 初始化投资组合，包含现金金额和股票持仓
    portfolio = {
        "cash": args.initial_cash,  # 初始现金金额
        "margin_requirement": args.margin_requirement,  # 初始保证金要求
        "margin_used": 0.0,  # 所有空头头寸的总保证金使用量
        "positions": {
            ticker: {
                "long": 0,  # 持有的多头股票数量
                "short": 0,  # 持有的空头股票数量
                "long_cost_basis": 0.0,  # 多头头寸的平均成本基础
                "short_cost_basis": 0.0,  # 卖空股票的平均价格
                "short_margin_used": 0.0,  # 该股票空头头寸使用的保证金金额
            }
            for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # 多头头寸的已实现收益
                "short": 0.0,  # 空头头寸的已实现收益
            }
            for ticker in tickers
        },
    }

    # 运行对冲基金系统
    result = run_hedge_fund(
        tickers=tickers,           # 股票代码列表
        start_date=start_date,     # 开始日期
        end_date=end_date,         # 结束日期
        portfolio=portfolio,       # 投资组合
        show_reasoning=args.show_reasoning,  # 是否显示推理过程
        selected_analysts=selected_analysts, # 选择的分析师
        model_name=model_name,     # 模型名称
        model_provider=model_provider,  # 模型提供商
    )

    # 打印交易输出结果
    print_trading_output(result)
