"""用户手动执行：python -B test/test_llm.py，仅发送一次OpenAI兼容接口请求。"""

import os
import sys
from time import perf_counter


def main():
    names = ("OPENAI_MODEL_NAME", "OPENAI_BASE_URL", "OPENAI_API_KEY")
    settings = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name in names if not settings[name]]
    if missing:
        print("请先export：" + "、".join(missing), file=sys.stderr)
        return 2

    model_name, base_url, api_key = (settings[name] for name in names)

    def redact(value):
        return str(value).replace(api_key, "***")

    print("模型：" + redact(model_name), flush=True)
    print("URL：" + redact(base_url), flush=True)
    started = perf_counter()
    try:
        from openai import OpenAI

        with OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0) as client:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "请只回复 OK。"}],
            )
        reply = response.choices[0].message.content if response.choices else None
        if not reply or not reply.strip():
            raise RuntimeError("接口未返回非空文本回复")
    except Exception as error:
        print(
            f"测试失败（{perf_counter() - started:.2f}秒）："
            f"{type(error).__name__}: {redact(error)}",
            file=sys.stderr,
        )
        return 1

    print(f"测试成功（{perf_counter() - started:.2f}秒）")
    print("回复：" + redact(reply.strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
