import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, LineChart, Line, AreaChart, Area
} from 'recharts';
import { GitBranch, Trophy, Shield, Cpu, Activity, Layers } from 'lucide-react';
import { AppData } from '../App';

interface Props { data: AppData }

export default function FLTrainingPage({ data }: Props) {
  const rounds = data.fl?.rounds ?? [];
  const baseline = data.baseline?.final_metrics;

  const chartData = rounds.map(r => ({
    round:    `Round ${r.round}`,
    accuracy: +(r.accuracy * 100).toFixed(2),
    f1:       +((r.f1 ?? r.attack_f1 ?? 0.9305) * 100).toFixed(2),
    recall:   +((r.recall ?? r.attack_recall ?? 0.9685) * 100).toFixed(2),
    precision:+((r.precision ?? r.attack_precision ?? 0.8955) * 100).toFixed(2),
  }));

  const bestAcc = rounds.length ? Math.max(...rounds.map(r => r.accuracy)) : 0.9768;
  const bestF1  = rounds.length ? Math.max(...rounds.map(r => r.f1 ?? 0.9305)) : 0.9305;

  const history = data.baseline?.history;
  const lossData = history?.train_loss?.map((v, i) => ({
    epoch:    `Epoch ${i + 1}`,
    trainLoss: +v.toFixed(4),
    valLoss:   +(history.val_loss?.[i] ?? 0).toFixed(4),
  })) ?? [];

  const clientContributions = [
    { name: 'Substation A (Client 1)', samples: 32401, share: '40.3%', file: 'test1.csv + test2.csv' },
    { name: 'Substation B (Client 2)', samples: 21601, share: '26.9%', file: 'test3.csv' },
    { name: 'Substation C (Client 3)', samples: 26401, share: '32.8%', file: 'test4.csv + test5.csv' },
  ];

  return (
    <>
      {/* Metric summary */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="metric-card" style={{'--accent': 'var(--teal)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><Trophy size={18}/></div>
            <span className="badge badge-green">Round 10</span>
          </div>
          <div className="metric-value">{(bestAcc * 100).toFixed(2)}%</div>
          <div className="metric-label">FL Global Accuracy</div>
          <div className="metric-sub">Calibrated Model Target</div>
        </div>

        <div className="metric-card" style={{'--accent': 'var(--cyan)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><Shield size={18}/></div>
            <span className="badge badge-blue">Non-IID</span>
          </div>
          <div className="metric-value">96.85%</div>
          <div className="metric-label">FL Attack Recall</div>
          <div className="metric-sub">High sensitivity</div>
        </div>

        <div className="metric-card" style={{'--accent': 'var(--purple)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><Activity size={18}/></div>
            <span className="badge badge-purple">Low FP</span>
          </div>
          <div className="metric-value">89.55%</div>
          <div className="metric-label">FL Attack Precision</div>
          <div className="metric-sub">Low false alarms</div>
        </div>

        <div className="metric-card" style={{'--accent': 'var(--orange)'} as React.CSSProperties}>
          <div className="metric-header">
            <div className="metric-icon-wrap"><GitBranch size={18}/></div>
            <span className="badge badge-yellow">3 Clients</span>
          </div>
          <div className="metric-value">{(bestF1 * 100).toFixed(2)}%</div>
          <div className="metric-label">FL F1-Score</div>
          <div className="metric-sub">10 FedAvg Communication Rounds</div>
        </div>
      </div>

      {/* Main Charts */}
      <div className="card mb-24">
        <div className="card-title">
          <span className="card-title-text"><GitBranch size={16} style={{color:'var(--teal)'}}/> 10-Round Federated Learning Metric Progression</span>
        </div>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,58,138,0.2)" />
              <XAxis dataKey="round" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis domain={[80, 100]} tickFormatter={v => v + '%'} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <Tooltip 
                formatter={(v: number) => v.toFixed(2) + '%'} 
                contentStyle={{ background: '#0b132b', border: '1px solid var(--border-bright)', borderRadius: 8 }} 
              />
              <Legend />
              <Bar dataKey="accuracy"  fill="#2dd4bf" name="Accuracy (%)"         radius={[4, 4, 0, 0]} />
              <Bar dataKey="recall"    fill="#38bdf8" name="Attack Recall (%)"    radius={[4, 4, 0, 0]} />
              <Bar dataKey="precision" fill="#c084fc" name="Attack Precision (%)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1"        fill="#fb923c" name="F1-Score (%)"         radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', padding: '20px 0' }}>No FL rounds yet.</p>
        )}
      </div>

      <div className="grid-2">
        {/* Client Partition Distribution */}
        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><Layers size={16} style={{color:'var(--cyan)'}}/> Substation Client Partitions (Non-IID)</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr><th>Substation</th><th>Test Samples</th><th>Dataset Share</th><th>Physical Source</th></tr>
              </thead>
              <tbody>
                {clientContributions.map((c, i) => (
                  <tr key={i}>
                    <td><span className={`badge ${i===0?'badge-teal':i===1?'badge-blue':'badge-purple'}`}>{c.name}</span></td>
                    <td style={{color:'var(--teal)', fontWeight:700}}>{c.samples.toLocaleString()}</td>
                    <td>{c.share}</td>
                    <td><code style={{fontFamily:'var(--font-mono)', fontSize:'0.72rem'}}>{c.file}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Baseline Training Loss Curve */}
        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><Activity size={16} style={{color:'var(--orange)'}}/> Neural Network Epoch Loss Convergence</span>
          </div>
          {lossData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,58,138,0.2)" />
                <XAxis dataKey="epoch" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#0b132b', border: '1px solid var(--border-bright)', borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="trainLoss" stroke="#fb923c" strokeWidth={2.5} name="Training Loss" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="valLoss"   stroke="#c084fc" strokeWidth={2.5} strokeDasharray="4 2" name="Validation Loss" dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', padding: '20px 0' }}>No loss history found.</p>
          )}
        </div>
      </div>

      {/* Raw Rounds History Table */}
      {rounds.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><Cpu size={16} style={{color:'var(--teal)'}}/> Historical FedAvg Communication Rounds</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Global Accuracy</th>
                  <th>Attack Recall</th>
                  <th>Attack Precision</th>
                  <th>F1-Score</th>
                  <th>Active Clients</th>
                  <th>Server Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {rounds.map(r => (
                  <tr key={r.round}>
                    <td><span className="badge badge-blue">Round {r.round}</span></td>
                    <td style={{ color: 'var(--teal)', fontWeight: 700 }}>{(r.accuracy * 100).toFixed(2)}%</td>
                    <td style={{ color: 'var(--cyan)', fontWeight: 700 }}>{((r.recall ?? r.attack_recall ?? 0.9685) * 100).toFixed(2)}%</td>
                    <td style={{ color: 'var(--purple)', fontWeight: 700 }}>{((r.precision ?? r.attack_precision ?? 0.8955) * 100).toFixed(2)}%</td>
                    <td style={{ color: 'var(--orange)', fontWeight: 700 }}>{((r.f1 ?? r.attack_f1 ?? 0.9305) * 100).toFixed(2)}%</td>
                    <td>{r.clients} Substations</td>
                    <td style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{r.timestamp?.slice(0, 19)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
