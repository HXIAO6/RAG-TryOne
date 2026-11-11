import time
import httpx


url = "https://httpbin.org/post"

data = {
    "question": "设备最高工作压力是多少？"
}


for attempt in range(3):

    try:
        response = httpx.post(
            url,
            json=data,
            timeout=5.0
        )

        response.raise_for_status()

        print("请求成功")
        print(response.json())

        break

    except httpx.TimeoutException:
        print(f"第 {attempt + 1} 次请求超时")

    except httpx.RequestError as e:
        print(f"第 {attempt + 1} 次请求失败:", e)

    time.sleep(1)