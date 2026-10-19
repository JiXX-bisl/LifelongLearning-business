from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.chatflow_agent.query import QueryService
from src.chatflow_agent.feedback import save_feedback
from src.chatflow_agent.runtime import BusinessRuntimeError
from src.project_paths import resolve_project_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/business/query.yaml"


def render_result(result: dict[str, Any]) -> None:
    """显示用户回复；开发信息默认折叠。"""
    st.markdown(result["answer"])

    if result["status"] != "completed":
        st.warning("本次查询未完整完成，请查看回复中的说明。")

    with st.expander("查看查询记录"):
        operations = result.get("operations", [])
        if not operations:
            st.caption("本次没有执行新的数据库操作。")

        for index, operation in enumerate(operations, start=1):
            st.markdown(f"**操作 {index}**")
            st.code(operation["sql"], language="sql")

            # 直接展示已有记录，保留列顺序、空值和完整性字段。
            st.json({
                key: value
                for key, value in operation.items()
                if key != "sql"
            })

    with st.expander("查看执行与调用统计"):
        st.json({
            "status": result["status"],
            "mode": result["mode"],
            "experiment_id": result["experiment_id"],
            "rounds": result["rounds"],
            "memory_version": result["memory_version"],
            "used_experience_ids": result["used_experience_ids"],
            "metrics": result.get("metrics", {}),
            "model_calls": result.get("model_calls", []),
        })
        st.caption(
            "Token 为 null 表示无法取得完整统计，不等于零。"
            "当前未配置价格，不估算调用金额。"
        )
        st.caption(f"会话记录：{result['record_path']}")

    record_path = resolve_project_path(result["record_path"])

    try:
        record_bytes = record_path.read_bytes()
    except OSError:
        st.warning("会话文件暂时无法读取，不能下载本次记录。")
    else:
        task_key = (
            f"{result['session_id']}_{result['turn_index']:04d}"
        )

        st.download_button(
            label="下载本次任务 JSON",
            data=record_bytes,
            file_name=f"query_{task_key}.json",
            mime="application/json",
            key=f"download_{task_key}",
            on_click="ignore",
            disabled=(
                st.session_state.get("pending_question") is not None
            ),
        )
        st.caption(
            "包含本次问题、上下文、查询结果和调用统计；"
            "对外分享前请注意其中的数据。"
        )

def render_feedback(result: dict[str, Any]) -> None:
    task_key = (
        f"{result['session_id']}_{result['turn_index']:04d}"
    )
    busy = (
        st.session_state.get("pending_question") is not None
    )
    labels = {
        "helpful": "有帮助",
        "problem": "有问题",
    }

    with st.expander("反馈这次回复"):
        with st.form(f"feedback_form_{task_key}"):
            rating = st.radio(
                "这次回复是否满足你的需要？",
                options=["helpful", "problem"],
                index=None,
                format_func=labels.get,
                horizontal=True,
                key=f"feedback_rating_{task_key}",
                disabled=busy,
            )
            comment = st.text_area(
                "补充说明（可选）",
                key=f"feedback_comment_{task_key}",
                disabled=busy,
            )
            submitted = st.form_submit_button(
                "保存 / 更新反馈",
                disabled=busy,
            )

        st.caption(
            "反馈仅用于后续改进，不自动改变经验库。"
        )

        if submitted:
            if rating is None:
                st.warning("请先选择“有帮助”或“有问题”。")
            else:
                try:
                    feedback_path = save_feedback(
                        result,
                        rating=rating,
                        comment=comment,
                    )
                except BusinessRuntimeError as error:
                    st.error(str(error))
                else:
                    st.success("反馈已保存；再次提交会更新同一条反馈。")
                    st.caption(f"反馈文件：{feedback_path}")

def main() -> None:
    st.set_page_config(
        page_title="企业查询助手",
        page_icon="💬",
        layout="wide",
    )

    state = st.session_state
    if "query_service" not in state:
        state.query_service = None
        state.query_turns = []
        state.pending_question = None

    busy = state.pending_question is not None

    st.title("企业查询助手")
    st.caption("基于虚拟企业数据库查询；历史经验仅用于参考方法。")

    with st.sidebar:
        st.subheader("运行设置")

        with st.form("query_settings"):
            mode = st.selectbox(
                "运行模式",
                options=["standard", "psu", "gsc"],
                disabled=busy,
            )
            experiment_id = st.text_input(
                "实验标识",
                value="q07_dev",
                disabled=busy,
            )
            apply_settings = st.form_submit_button(
                "启动 / 应用设置",
                disabled=busy,
            )

        st.caption("应用设置会关闭旧服务并开始新对话，不删除经验。")

        if apply_settings:
            try:
                if state.query_service is not None:
                    state.query_service.close()
                    state.query_service = None

                state.query_turns = []
                state.query_service = QueryService(
                    CONFIG_PATH,
                    mode=mode,
                    experiment_id=experiment_id.strip(),
                )
            except BusinessRuntimeError as error:
                st.error(str(error))
            except Exception:
                st.error("服务启动失败，请检查本地环境和连接配置。")
            else:
                st.rerun()

        service = state.query_service
        if service is not None:
            st.success("服务已启动")
            st.caption(f"当前模式：{service.learning['mode']}")
            st.caption(
                f"当前实验：{service.learning['experiment_id']}"
            )

        if st.button(
            "新对话",
            disabled=busy or service is None,
        ):
            service.reset()
            state.query_turns = []
            st.rerun()

        if st.button(
            "关闭服务",
            disabled=busy or service is None,
        ):
            try:
                service.close()
            except Exception:
                st.error("关闭连接时发生异常，请停止并重新启动网页进程。")
            else:
                state.query_service = None
                state.query_turns = []
                st.rerun()

    # 渲染当前页面会话中的历史问答。
    for turn in state.query_turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])

        with st.chat_message("assistant"):
            if "result" in turn:
                render_result(turn["result"])
                render_feedback(turn["result"])
            else:
                st.error(turn["error"])

    service = state.query_service

    if service is None:
        st.info("请先在左侧启动服务。启动预检不会请求模型。")

    question = st.chat_input(
        "请输入企业查询问题",
        disabled=busy or service is None,
    )

    if question is not None and question.strip():
        state.pending_question = question.strip()

        # 先重绘页面，禁用操作按钮，再执行耗时查询。
        st.rerun()

    if state.pending_question is None:
        return

    pending = state.pending_question

    with st.chat_message("user"):
        st.markdown(pending)

    with st.chat_message("assistant"):
        with st.status("正在处理……", expanded=False) as progress:
            def show_progress(message: str) -> None:
                progress.update(label=message)

            try:
                result = service.query(
                    pending,
                    on_progress=show_progress,
                )
            except BusinessRuntimeError as error:
                turn = {
                    "question": pending,
                    "error": str(error),
                }
            except Exception:
                turn = {
                    "question": pending,
                    "error": (
                        "本次处理发生异常。请检查本地服务，"
                        "不要直接假定任务已成功或记录已保存。"
                    ),
                }
            else:
                turn = {
                    "question": pending,
                    "result": result,
                }
            finally:
                state.pending_question = None

            state.query_turns.append(turn)

    # 展示结果并恢复输入，不重复执行刚才的问题。
    st.rerun()
