# Copyright 2024 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import time

import aiohttp


async def fetch(session, url, data):
    try:
        async with session.get(url, json=data) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientConnectionError as e:
        print(e)


async def request(urls, data):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url, data) for url in urls]
        return await asyncio.gather(*tasks)


def parse_host_ranks(host_list, rank_list, port, dry_run=False):
    combined_hosts = []
    for host, rank in zip(host_list, rank_list):
        if not rank:
            combined_hosts.append(f"{host}:{port}")
            continue
        if "-" not in rank:
            combined_hosts.append(f"{host}-{rank}:{port}")
            continue
        start, end = map(int, rank.split("-"))
        for r in range(start, end + 1):
            combined_hosts.append(f"{host}-{r}:{port}")
    if dry_run:
        return combined_hosts
    return [f"http://{host}/HostingService/DumpKernelTrace" for host in combined_hosts]


def AES128_CBC(text):
    from base64 import b64encode

    from Crypto.Cipher import AES
    from Crypto.Util.Padding import pad

    iv = "xpu_timer"
    password = "xpu_timer"
    iv = iv.ljust(16, "\0").encode("utf-8")
    password = password.ljust(16, "\0").encode("utf-8")
    text = pad(text.encode("utf-8"), AES.block_size)

    cipher = AES.new(password, AES.MODE_CBC, iv)
    cipher_text = cipher.encrypt(text)
    return b64encode(cipher_text).decode("utf-8")


def main():
    # curr=`date +%s`
    # curl -H 'Content-Type: application/json' \
    #       -d "{\"dump_path\":\"/root/cc/dd/ee\", \"dump_count\": 110, \"dump_time\": $((curr+3))}" \
    #       127.0.0.1:18888/HostingService/DumpKernelTrace
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", required=True, help="Specify the host")
    parser.add_argument("--rank", action="append", required=True, help="Specify the host rank range")
    parser.add_argument("--port", type=int, default=18888, help="Specify the port on host")
    parser.add_argument("--dump-path", type=str, default="/root/timeline", help="Specify dump path")
    parser.add_argument("--dump-count", type=int, default=1000, help="Specify how many events to dump")
    parser.add_argument("--delay", type=int, default=5, help="Specify when dump after request")
    parser.add_argument("--reset", action="store_true", help="Specify reset dump flag")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--no-nccl", action="store_true", help="Disable nccl trace")
    parser.add_argument("--no-matmul", action="store_true", help="Disable matmul(fa) trace")
    parser.add_argument("--no-memory", action="store_true", help="Disable memory trace")
    parser.add_argument("--oss-path", type=str, default="", help="Specify oss dump path")
    parser.add_argument("--oss-ak", type=str, default="", help="Specify oss ak")
    parser.add_argument("--oss-sk", type=str, default="", help="Specify oss sk")
    parser.add_argument("--oss-endpoint", type=str, default="", help="Specify oss endpoint")

    args = parser.parse_args()

    if len(args.host) != len(args.rank):
        parser.error("--host and --rank must be provided in pairs")

    dump_kernel_type = 7  # [00][11] // first bits is matmul, second bits is nccl
    if args.no_nccl:
        dump_kernel_type -= 2
        print("Disable nccl traces")
    if args.no_matmul:
        dump_kernel_type -= 1
        print("Disable matmul traces")
    if args.no_memory:
        dump_kernel_type -= 4
        print("Disable memory traces")
    if dump_kernel_type == 0:
        raise ValueError("No Kernel to trace")
    combined_hosts = parse_host_ranks(args.host, args.rank, args.port, args.dry_run)
    now = int(time.time())
    print(f"dumping to {args.dump_path}, with count {args.dump_count}")
    data = {
        "dump_path": args.dump_path,
        "dump_time": now + args.delay,
        "dump_count": args.dump_count,
        "reset": args.reset,
        "dump_kernel_type": dump_kernel_type,
    }
    if args.oss_path and args.oss_ak and args.oss_sk and args.oss_endpoint:
        data["oss_args"] = {
            "oss_ak": AES128_CBC(args.oss_ak),
            "oss_sk": AES128_CBC(args.oss_sk),
            "oss_endpoint": args.oss_endpoint,
            "oss_path": args.oss_path,
        }
    print(data)
    if args.dry_run:
        print(f"dump host {combined_hosts}")
        print(f"other data {data}")
        return
    for i in asyncio.run(request(combined_hosts, data)):
        print(i)


if __name__ == "__main__":
    main()
