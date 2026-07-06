#!/usr/bin/env python3
"""
ids_controller.py — Fixed 15-feature version
/home/mininet/Desktop/topology_wise/
"""
import os, csv, json, time, pickle, logging, threading, collections
from datetime import datetime
import numpy as np
from ryu.base               import app_manager
from ryu.controller         import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto            import ofproto_v1_3
from ryu.lib.packet         import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.lib                import hub

BASE  = "/home/mininet/Desktop/topology_wise"
MODEL = os.path.join(BASE, "final_xgb_model_bundle.pkl")
LOG   = os.path.join(BASE, "topology_performance_logs.csv")
META  = "/tmp/sdn_topo_meta.json"

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
L = logging.getLogger("IDS")

LABEL_MAP = {
    0:"BENIGN",1:"DoS Hulk",2:"DoS Slowloris",3:"DoS GoldenEye",
    4:"DDoS",5:"PortScan",6:"FTP-Patator",7:"SSH-Patator",
    8:"Heartbleed",9:"Bot",10:"Web-SQLi",11:"Web-XSS",
    12:"Infiltration",13:"Web-BruteForce",14:"DoS Slowhttptest",
}
MIN_PKTS = 3
MIN_DUR  = 0.1

class Flow:
    def __init__(self,src,dst,proto):
        now=time.time()
        self.src=src;self.dst=dst;self.proto=proto;self.t0=now
        self.fp=0;self.bp=0;self.fb=0;self.bb=0
        self.fl=[];self.bl=[];self.fi=[];self.bi=[]
        self.lft=now;self.lbt=now;self.fh=0;self.bh=0

    def fwd(self,plen,hdr=20):
        now=time.time()
        if self.fp>0: self.fi.append((now-self.lft)*1000)
        self.lft=now;self.fp+=1;self.fb+=plen;self.fl.append(plen);self.fh+=hdr

    def bwd(self,plen,hdr=20):
        now=time.time()
        if self.bp>0: self.bi.append((now-self.lbt)*1000)
        self.lbt=now;self.bp+=1;self.bb+=plen;self.bl.append(plen);self.bh+=hdr

    def dur(self): return max(time.time()-self.t0,1e-6)

    def vec(self, n_feat):
        d=self.dur()
        fm=float(np.mean(self.fl)) if self.fl else 0.0
        bm=float(np.mean(self.bl)) if self.bl else 0.0
        fim=float(np.mean(self.fi)) if self.fi else 0.0
        bim=float(np.mean(self.bi)) if self.bi else 0.0
        fis=float(np.std(self.fi))  if self.fi else 0.0
        bis=float(np.std(self.bi))  if self.bi else 0.0
        fps=self.fp/d; bps_=self.bp/d
        tbps=(self.fb+self.bb)/d; tpps=(self.fp+self.bp)/d

        # Full 15-feature vector
        all_feats = [d,self.fp,self.bp,fps,bps_,fm,bm,
                     fim,bim,fis,bis,float(self.fh),float(self.bh),
                     tbps,tpps]
        # Trim or pad to exactly n_feat
        all_feats = all_feats[:n_feat]
        while len(all_feats) < n_feat:
            all_feats.append(0.0)
        return np.array(all_feats,dtype=np.float32).reshape(1,-1)

HDR=["topology","timestamp","src_ip","dst_ip","proto",
     "predicted_label","predicted_label_id","confidence_pct",
     "detection_delay_ms","action","flow_duration_s",
     "fwd_pkts","bwd_pkts","fwd_bytes","bwd_bytes"]

class CSVW:
    def __init__(self,path):
        self.path=path;self._lk=threading.Lock()
        if not os.path.exists(path) or os.path.getsize(path)==0:
            with open(path,"w",newline="") as f:
                csv.DictWriter(f,fieldnames=HDR).writeheader()
        L.info(f"CSV -> {path}")

    def write(self,row):
        with self._lk:
            with open(self.path,"a",newline="") as f:
                csv.DictWriter(f,fieldnames=HDR,extrasaction="ignore").writerow(row)

class IDSController(app_manager.RyuApp):
    OFP_VERSIONS=[ofproto_v1_3.OFP_VERSION]

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        L.info(f"Loading {MODEL}")
        with open(MODEL,"rb") as f: b=pickle.load(f)
        self.model  = b.get("model") or b.get("classifier")
        self.scaler = b.get("scaler")
        self.encoder= b.get("label_encoder")
        self.n_feat = int(getattr(self.model,"n_features_in_",15))
        L.info(f"Model OK — classes={getattr(self.model,'n_classes_','?')}  features={self.n_feat}")
        self.mac_to_port={}; self.datapaths={}
        self.flows=collections.defaultdict(dict)
        self.blocked=set(); self._lk=threading.Lock()
        self.csv=CSVW(LOG)
        self._mon=hub.spawn(self._loop)
        L.info("IDSController ready — polling every 2 s")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures,CONFIG_DISPATCHER)
    def sw_feat(self,ev):
        dp=ev.msg.datapath;ofp=dp.ofproto;p=dp.ofproto_parser
        self.datapaths[dp.id]=dp
        L.info(f"Switch connected dpid={dp.id:#018x}")
        self._add_flow(dp,0,p.OFPMatch(),
            [p.OFPActionOutput(ofp.OFPP_CONTROLLER,ofp.OFPCML_NO_BUFFER)])

    @set_ev_cls(ofp_event.EventOFPPacketIn,MAIN_DISPATCHER)
    def pkt_in(self,ev):
        msg=ev.msg;dp=msg.datapath;ofp=dp.ofproto;p=dp.ofproto_parser
        in_port=msg.match["in_port"]
        pkt=packet.Packet(msg.data)
        eth=pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype==0x88cc: return
        self.mac_to_port.setdefault(dp.id,{})[eth.src]=in_port
        out=self.mac_to_port[dp.id].get(eth.dst,ofp.OFPP_FLOOD)
        ip4=pkt.get_protocol(ipv4.ipv4)
        if ip4:
            src=ip4.src;dst=ip4.dst;proto=ip4.proto
            if src in self.blocked: return
            t4=pkt.get_protocol(tcp.tcp)
            hdr=t4.offset*4 if t4 else 8
            key=(src,dst,proto)
            with self._lk:
                if key not in self.flows[dp.id]:
                    self.flows[dp.id][key]=Flow(src,dst,proto)
                rec=self.flows[dp.id][key]
                if dst=="10.0.0.5": rec.fwd(len(msg.data),hdr)
                else:               rec.bwd(len(msg.data),hdr)
        acts=[p.OFPActionOutput(out)]
        if out!=ofp.OFPP_FLOOD:
            self._add_flow(dp,1,p.OFPMatch(in_port=in_port,eth_dst=eth.dst),acts)
        data=msg.data if msg.buffer_id==ofp.OFP_NO_BUFFER else None
        dp.send_msg(p.OFPPacketOut(datapath=dp,buffer_id=msg.buffer_id,
                                   in_port=in_port,actions=acts,data=data))

    def _loop(self):
        while True:
            hub.sleep(2)
            self._classify()

    def _classify(self):
        topo=self._topo()
        with self._lk:
            snap={d:dict(fl) for d,fl in self.flows.items()}
        for dpid,flows in snap.items():
            dp=self.datapaths.get(dpid)
            for key,rec in flows.items():
                src,dst,proto=key
                if rec.fp+rec.bp<MIN_PKTS: continue
                if rec.dur()<MIN_DUR: continue
                try:
                    t0=time.time()
                    X=rec.vec(self.n_feat)
                    if self.scaler: X=self.scaler.transform(X)
                    pid=int(self.model.predict(X)[0])
                    proba=self.model.predict_proba(X)[0]
                    conf=float(np.max(proba))*100
                    label=(str(self.encoder.inverse_transform([pid])[0])
                           if self.encoder else LABEL_MAP.get(pid,f"C{pid}"))
                    delay=(time.time()-t0)*1000
                    is_atk=(pid!=0)
                    action="DROP" if is_atk else "PERMIT"
                    if is_atk:
                        L.warning(f"[ATTACK] {src}->{dst} {label} {conf:.1f}% {action}")
                        if dp: self._drop(dp,src)
                        self.blocked.add(src)
                    else:
                        L.info(f"[BENIGN] {src}->{dst} conf={conf:.1f}% {delay:.1f}ms")
                    self.csv.write({"topology":topo,
                        "timestamp":datetime.utcnow().isoformat(),
                        "src_ip":src,"dst_ip":dst,"proto":proto,
                        "predicted_label":label,"predicted_label_id":pid,
                        "confidence_pct":round(conf,2),
                        "detection_delay_ms":round(delay,3),
                        "action":action,"flow_duration_s":round(rec.dur(),3),
                        "fwd_pkts":rec.fp,"bwd_pkts":rec.bp,
                        "fwd_bytes":rec.fb,"bwd_bytes":rec.bb})
                except Exception as e:
                    L.error(f"Error {src}->{dst}: {e}")
                finally:
                    with self._lk: self.flows[dpid].pop(key,None)

    def _add_flow(self,dp,pri,match,acts,idle=0,hard=0):
        p=dp.ofproto_parser;ofp=dp.ofproto
        dp.send_msg(p.OFPFlowMod(datapath=dp,priority=pri,
            idle_timeout=idle,hard_timeout=hard,match=match,
            instructions=[p.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS,acts)]))

    def _drop(self,dp,src,pri=100,hard=300):
        p=dp.ofproto_parser
        dp.send_msg(p.OFPFlowMod(datapath=dp,priority=pri,hard_timeout=hard,
            match=p.OFPMatch(eth_type=0x0800,ipv4_src=src),instructions=[]))
        L.info(f"[IPS] DROP -> {src}")

    @staticmethod
    def _topo():
        try:
            with open(META) as f: return json.load(f).get("topology","unknown")
        except: return "unknown"
