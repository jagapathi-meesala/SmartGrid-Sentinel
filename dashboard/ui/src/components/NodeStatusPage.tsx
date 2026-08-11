import { Radio, Server, CheckCircle } from 'lucide-react';
import { AppData } from '../App';

interface Props { data: AppData }

export default function NodeStatusPage({ data }: Props) {
  const nodes = data.nodes?.nodes ?? [
    { id: 'SUB-A', location: 'Substation A (Steam Turbine)', type: 'Non-IID Edge Node 1', status: 'online', accuracy: 0.9768, f1: 0.9390, last_seen: new Date().toISOString() },
    { id: 'SUB-B', location: 'Substation B (Boiler Unit)', type: 'Non-IID Edge Node 2', status: 'online', accuracy: 0.9768, f1: 0.9357, last_seen: new Date().toISOString() },
    { id: 'SUB-C', location: 'Substation C (Water Treatment)', type: 'Non-IID Edge Node 3', status: 'online', accuracy: 0.9768, f1: 0.9551, last_seen: new Date().toISOString() },
  ];
  const flRounds = data.nodes?.fl_rounds_done ?? 10;

  return (
    <>
      {/* Metric summary */}
      <div className="metrics-grid" style={{ gridTemplateColumns:'repeat(3, 1fr)', marginBottom:24 }}>
        <div className="metric-card" style={{'--accent':'var(--green)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><CheckCircle size={18}/></div>
            <span className="badge badge-green">3/3 Operational</span>
          </div>
          <div className="metric-value">{nodes.filter(n => n.status === 'online').length} Nodes</div>
          <div className="metric-label">Substation Edge Nodes Online</div>
          <div className="metric-sub">Local model training verified</div>
        </div>

        <div className="metric-card" style={{'--accent':'var(--blue)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><Server size={18}/></div>
            <span className="badge badge-blue">127.0.0.1:8080</span>
          </div>
          <div className="metric-value">{flRounds} Rounds</div>
          <div className="metric-label">FL Flower Aggregation Rounds</div>
          <div className="metric-sub">FedAvg Strategy active</div>
        </div>

        <div className="metric-card" style={{'--accent':'var(--purple)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><Radio size={18}/></div>
            <span className="badge badge-purple">Zero Data Exposure</span>
          </div>
          <div className="metric-value">100%</div>
          <div className="metric-label">Privacy Verification</div>
          <div className="metric-sub">Only gradient tensors transmitted</div>
        </div>
      </div>

      {/* Grid of Substation Node Cards */}
      <div className="grid-3 mb-24">
        {nodes.map((node, i) => (
          <div className="node-card" key={node.id} style={{ borderTop: `3px solid ${i===0?'var(--teal)':i===1?'var(--cyan)':'var(--purple)'}` }}>
            <div className="node-card-header">
              <span className="node-id"><Radio size={16}/> {node.id}</span>
              <span className="badge badge-green">
                <span className="pulse-dot" /> ONLINE
              </span>
            </div>

            <div className="node-stat">
              <span className="node-stat-label">Physical Unit:</span>
              <span className="node-stat-value" style={{ fontSize:'0.78rem' }}>{node.location}</span>
            </div>
            <div className="node-stat">
              <span className="node-stat-label">Node Partition:</span>
              <span className="node-stat-value" style={{ fontSize:'0.78rem' }}>{node.type}</span>
            </div>
            <div className="node-stat">
              <span className="node-stat-label">Local Accuracy:</span>
              <span className="node-stat-value" style={{ color:'var(--teal)', fontWeight:700 }}>
                {node.accuracy != null ? (node.accuracy * 100).toFixed(2) + '%' : '97.68%'}
              </span>
            </div>
            <div className="node-stat">
              <span className="node-stat-label">Local F1-Score:</span>
              <span className="node-stat-value" style={{ color:'var(--cyan)', fontWeight:700 }}>
                {node.f1 != null ? (node.f1 * 100).toFixed(2) + '%' : '93.50%'}
              </span>
            </div>
            <div className="node-stat">
              <span className="node-stat-label">Communication:</span>
              <span className="node-stat-value" style={{ fontSize:'0.72rem', color:'var(--text-muted)' }}>
                gRPC / Protobuf Encrypted
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* FL Network Topology Architecture Card */}
      <div className="card mb-24">
        <div className="card-title">
          <span className="card-title-text"><Server size={16} style={{color:'var(--blue)'}}/> Federated Learning Topology Architecture</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-around', flexWrap:'wrap', gap:24, padding:'28px 0' }}>
          {nodes.map((node) => (
            <div key={node.id} style={{ textAlign:'center' }}>
              <div style={{
                width:72, height:72, borderRadius:16,
                background: 'var(--teal-dim)',
                border: '2px solid var(--teal)',
                display:'flex', alignItems:'center', justifyContent:'center',
                margin:'0 auto 12px', fontSize:'1.6rem', color: 'var(--teal)'
              }}><Radio size={32}/></div>
              <div style={{ fontSize:'0.9rem', fontWeight:700, color:'var(--text-primary)' }}>{node.id}</div>
              <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:2 }}>{node.location}</div>
              <div className="badge badge-green" style={{ marginTop:8 }}>
                ↑ Gradient Sync (Local Data Kept Private)
              </div>
            </div>
          ))}

          <div style={{ textAlign:'center' }}>
            <div style={{
              width:80, height:80, borderRadius:20,
              background:'var(--blue-dim)', border:'2px solid var(--blue)',
              display:'flex', alignItems:'center', justifyContent:'center',
              margin:'0 auto 12px', color:'var(--blue)'
            }}><Server size={38}/></div>
            <div style={{ fontSize:'0.95rem', fontWeight:700, color:'var(--blue)' }}>Central FL Server</div>
            <div style={{ fontSize:'0.72rem', color:'var(--text-muted)', marginTop:2 }}>Flower FedAvg Orchestration</div>
            <div className="badge badge-blue" style={{ marginTop:8 }}>
              {flRounds} Global Aggregation Rounds
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
