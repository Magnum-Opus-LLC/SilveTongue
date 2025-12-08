# proxy_manager.py
import requests
from typing import List


def fetch_proxy_list() -> List[str]:
    proxy_list_url = "https://spys.me/proxy.txt"
    response = requests.get(proxy_list_url)
    text = response.text
    proxies = []
    for line in text.splitlines():
        parts = line.split()
        if parts:
            ip_port = parts[0]
            if ":" in ip_port:
                proxies.append(ip_port)
    return proxies
