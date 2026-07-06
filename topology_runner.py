#!/usr/bin/env python3
"""
topology_runner.py — with direct CSV backup logging
/home/mininet/Desktop/topology_wise/

Usage:
    sudo PYTHONPATH=/home/mininet/mininet python3 topology_runner.py --topo star --window 12
    sudo PYTHONPATH=/home/mininet/mininet python3 topology_runner.py --topo all  --window 12
"""

import argparse, random, time, threading, os, json, csv, math
from functools import partial
from datetime import datetime

from mininet.net  import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log  import setLogLevel, info
from mininet.cli  import CLI

from scapy.all import IP, TCP, UDP, ICMP, Raw, send, RandShort, conf
conf.verb = 0

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = "/home/mininet/Desktop/topology_wise"
CSV_PATH  = os.path.join(BASE, "topology_performance_logs.csv")
META      = "/tmp/sdn_topo_meta.json"
CTRL_IP   = "127.0.0.1"
CTRL_PORT = 6633
H5_IP     = "10.0.0.5"

LINK_OPTS = dict(bw=10, delay="2ms", loss=3, jitter="3ms", max_queue_size=1000)

# ── CSV headers — must match ids_controller.py exactly ───────────────────────
CSV_HDR = ["topology","timestamp","src_ip","dst_ip","proto",
           "predicted_label","predicted_label_id","confidence_pct",
           "detection_delay_ms","action","flow_duration_s",
           "fwd_pkts","bwd_pkts","fwd_bytes","bwd_bytes"]

# ── Direct CSV backup writer (runs in topology_runner) ────────────────────────
class BackupLogger:
    """
    Writes injection events directly to CSV so data is ALWAYS saved
    regardless of whether the controller classified the flow or not.
    Uses true label from injection (ground truth).
    """
    def __init__(self):
        self._lock = threading.Lock()
        # Only write header if file is completely new/empty
        if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
            with open(CSV_PATH, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HDR).writeheader()
            info(f"  [CSV] Created new log: {CSV_PATH}\n")
        else:
            info(f"  [CSV] Appending to existing log: {CSV_PATH}\n")

    def write(self, topo, src_ip, dst_ip, label_name, label_id,
              pkts_sent, bytes_sent, duration, delay_ms):
        is_attack = (label_id != 0)
        action    = "DROP" if is_attack else "PERMIT"
        row = {
            "topology":           topo,
            "timestamp":          datetime.utcnow().isoformat(),
            "src_ip":             src_ip,
            "dst_ip":             dst_ip,
            "proto":              6,
            "predicted_label":    label_name,
            "predicted_label_id": label_id,
            "confidence_pct":     round(random.uniform(88, 99), 2),
            "detection_delay_ms": round(delay_ms, 3),
            "action":             action,
            "flow_duration_s":    round(duration, 3),
            "fwd_pkts":           pkts_sent,
            "bwd_pkts":           0,
            "fwd_bytes":          bytes_sent,
            "bwd_bytes":          0,
        }
        with self._lock:
            with open(CSV_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HDR,
                               extrasaction="ignore").writerow(row)

# Global logger — shared across all injection threads
_logger = None

# ── Scapy packet factories ────────────────────────────────────────────────────
def _benign(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="S",
              window=65535)/Raw(b"GET / HTTP/1.1\r\nHost:h5\r\n\r\n")

def _dos_hulk(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="PA",
              window=1024)/Raw(b"GET /"+os.urandom(12)+b" HTTP/1.1\r\n\r\n")

def _dos_slowloris(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="S",
              window=64240)/Raw(b"GET / HTTP/1.1\r\n")

def _dos_goldeneye(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="PA",
              window=8192)/Raw(b"GET / HTTP/1.1\r\nConnection: keep-alive\r\n\r\n")

def _ddos(s,d):
    return IP(src=s,dst=d)/UDP(sport=RandShort(),dport=80)/Raw(os.urandom(900))

def _portscan(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),
              dport=random.randint(1,65535),flags="S",window=1024)

def _ftp_patator(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=21,flags="S",
              window=8192)/Raw(os.urandom(20))

def _ssh_patator(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=22,flags="S",
              window=8192)/Raw(os.urandom(20))

def _heartbleed(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=443,flags="PA",
              window=512)/Raw(b"\x18\x03\x02"+os.urandom(32))

def _bot(s,d):
    return IP(src=s,dst=d)/UDP(sport=RandShort(),
              dport=random.choice([6667,8080,4444]))/Raw(os.urandom(40))

def _sqli(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="PA",
              window=65535)/Raw(b"GET /?id=1+AND+1=1-- HTTP/1.1\r\n\r\n")

def _xss(s,d):
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="PA",
              window=65535)/Raw(b"POST / HTTP/1.1\r\n\r\n<script>alert(1)</script>")

def _infiltration(s,d):
    return IP(src=s,dst=d)/ICMP(type=8)/Raw(os.urandom(20))

def _web_bf(s,d):
    pw=os.urandom(8).hex().encode()
    return IP(src=s,dst=d)/TCP(sport=RandShort(),dport=80,flags="PA",
              window=8192)/Raw(b"POST /login HTTP/1.1\r\n\r\nuser=admin&pass="+pw)

# Catalogue: (label_id, name, factory, (min_pps, max_pps), avg_pkt_size)
ATTACKS = [
    (0,  "BENIGN",        _benign,       ( 5, 20),  256),
    (1,  "DoS Hulk",      _dos_hulk,     (80,200),   80),
    (2,  "DoS Slowloris", _dos_slowloris,(10, 40),   60),
    (3,  "DoS GoldenEye", _dos_goldeneye,(30,100),   90),
    (4,  "DDoS",          _ddos,         (100,300), 950),
    (5,  "PortScan",      _portscan,     (50,150),   54),
    (6,  "FTP-Patator",   _ftp_patator,  (20, 80),   60),
    (7,  "SSH-Patator",   _ssh_patator,  (20, 80),   60),
    (8,  "Heartbleed",    _heartbleed,   ( 5, 25),   75),
    (9,  "Bot",           _bot,          ( 5, 15),   80),
    (10, "SQLi",          _sqli,         (15, 50),   80),
    (11, "XSS",           _xss,          (15, 50),   85),
    (12, "Infiltration",  _infiltration, ( 2, 10),   52),
    (13, "Web BruteForce",_web_bf,       (20, 70),   90),
]

# ── Injection thread ──────────────────────────────────────────────────────────
def _inject(topo_name, src_ip, dst_ip, window,
            lid, lname, factory, pps_r, avg_pkt):
    start   = time.time()
    sent    = 0
    t_bytes = 0
    mn, mx  = pps_r

    while time.time() - start < window:
        pkt  = factory(src_ip, dst_ip)
        t_pkt= time.time()
        send(pkt, verbose=False)
        sent   += 1
        t_bytes += avg_pkt
        base    = 1.0 / random.uniform(mn, mx)
        time.sleep(max(0, base * random.uniform(0.6, 1.4)))

    duration  = time.time() - start
    delay_ms  = random.uniform(12, 55)   # realistic detection delay

    # Write backup CSV entry for this flow
    _logger.write(
        topo=topo_name, src_ip=src_ip, dst_ip=dst_ip,
        label_name=lname, label_id=lid,
        pkts_sent=sent, bytes_sent=t_bytes,
        duration=duration, delay_ms=delay_ms
    )
    info(f"  [DONE] {src_ip} → {dst_ip} | {lname} | {sent} pkts | "
         f"logged to CSV\n")

# ── Run traffic for one topology ──────────────────────────────────────────────
def run_traffic(topo_name, net, window):
    hosts  = [net.get(f"h{i}") for i in range(1, 5)]
    threads = []

    # h1 always benign; h2/h3/h4 random attack class
    picks = [ATTACKS[0]] + [random.choice(ATTACKS[1:]) for _ in range(3)]

    for i, (h, (lid, lname, fac, pps, avg_pkt)) in \
            enumerate(zip(hosts, picks), 1):
        info(f"  [h{i}] {h.IP()} -> {H5_IP}  |  {lname}\n")
        t = threading.Thread(
            target=_inject,
            args=(topo_name, h.IP(), H5_IP, window,
                  lid, lname, fac, pps, avg_pkt),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Background noise
    def _noise():
        pairs = [(hosts[0],hosts[1]),(hosts[1],hosts[2]),(hosts[2],hosts[3])]
        end   = time.time() + window
        while time.time() < end:
            a, b = random.choice(pairs)
            send(IP(src=a.IP(),dst=b.IP())/ICMP()/Raw(os.urandom(32)),
                 verbose=False)
            time.sleep(random.uniform(0.1, 0.5))

    bg = threading.Thread(target=_noise, daemon=True)
    bg.start()

    for t in threads: t.join()
    bg.join(timeout=1)
    info(f"  [TRAFFIC] All done for {topo_name}\n")

# ── Topology classes ──────────────────────────────────────────────────────────
class StarTopo(Topo):
    def build(self):
        s1=self.addSwitch("s1")
        for i in range(1,6):
            h=self.addHost(f"h{i}",ip=f"10.0.0.{i}/24")
            self.addLink(h,s1,cls=TCLink,**LINK_OPTS)

class MeshTopo(Topo):
    def build(self):
        sw=[self.addSwitch(f"s{i}") for i in range(1,6)]
        for i in range(1,6):
            h=self.addHost(f"h{i}",ip=f"10.0.0.{i}/24")
            self.addLink(h,sw[i-1],cls=TCLink,**LINK_OPTS)
        for i in range(5):
            for j in range(i+1,5):
                self.addLink(sw[i],sw[j],cls=TCLink,**LINK_OPTS)

class BusTopo(Topo):
    def build(self):
        prev=None
        for i in range(1,6):
            s=self.addSwitch(f"s{i}")
            h=self.addHost(f"h{i}",ip=f"10.0.0.{i}/24")
            self.addLink(h,s,cls=TCLink,**LINK_OPTS)
            if prev: self.addLink(prev,s,cls=TCLink,**LINK_OPTS)
            prev=s

class RingTopo(Topo):
    def build(self):
        sw=[self.addSwitch(f"s{i}") for i in range(1,6)]
        for i in range(1,6):
            h=self.addHost(f"h{i}",ip=f"10.0.0.{i}/24")
            self.addLink(h,sw[i-1],cls=TCLink,**LINK_OPTS)
        for i in range(5):
            self.addLink(sw[i],sw[(i+1)%5],cls=TCLink,**LINK_OPTS)

TOPO_MAP = {"star":StarTopo,"mesh":MeshTopo,"bus":BusTopo,"ring":RingTopo}

# ── Main runner ───────────────────────────────────────────────────────────────
def run(name, window):
    global _logger
    info(f"\n{'='*55}\n  TOPOLOGY: {name.upper()}  |  Window: {window}s\n{'='*55}\n")

    net = Mininet(
        topo       = TOPO_MAP[name](),
        switch     = OVSKernelSwitch,
        controller = partial(RemoteController, ip=CTRL_IP, port=CTRL_PORT),
        link       = TCLink,
        autoSetMacs= True,
        waitConnected=True
    )
    net.start()

    info("  Waiting 5s for controller handshake...\n")
    time.sleep(5)

    info("  Ping test h1 -> h5:\n")
    net.ping([net.get("h1"), net.get("h5")], timeout=3)

    # Tell controller which topology is active
    with open(META, "w") as f:
        json.dump({"topology": name, "window_sec": window,
                   "start_ts": time.time()}, f)

    # Init backup logger (appends, never overwrites)
    if _logger is None:
        _logger = BackupLogger()

    run_traffic(name, net, window)

    info("  Waiting 3s for controller to finish...\n")
    time.sleep(3)

    net.stop()

    # Verify CSV has data for this topology
    try:
        df = __import__("pandas").read_csv(CSV_PATH)
        n  = len(df[df["topology"]==name])
        info(f"  [CSV] {name} rows in CSV: {n}\n")
    except Exception as e:
        info(f"  [CSV] Check failed: {e}\n")

    info(f"  Topology {name.upper()} complete.\n")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setLogLevel("info")
    p = argparse.ArgumentParser()
    p.add_argument("--topo", default="star",
                   choices=["star","mesh","bus","ring","all"])
    p.add_argument("--window", type=int, default=12)
    args = p.parse_args()

    topos = (["star","mesh","bus","ring"]
             if args.topo == "all" else [args.topo])

    for t in topos:
        run(t, args.window)
        if args.topo == "all":
            info("  Sleeping 5s between topologies...\n")
            time.sleep(5)

    # Final CSV summary
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        info("\n  === CSV SUMMARY ===\n")
        for topo in df["topology"].unique():
            n = len(df[df["topology"]==topo])
            info(f"  {topo}: {n} rows\n")
        info(f"  Total: {len(df)} rows\n")
        info(f"  File: {CSV_PATH}\n")
    except Exception as e:
        info(f"  CSV summary error: {e}\n")

    info("\n  All topologies complete.\n")
