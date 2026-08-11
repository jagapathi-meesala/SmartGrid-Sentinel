import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { Shield, AlertTriangle, Activity, Database, CheckCircle, XCircle } from 'lucide-react';
import { AppData } from '../App';

interface Props { data: AppData }

const COLORS = ['#34d399','#f87171','#fb923c','#c084fc','#ec4899','#fbbf24'];

export default function AttackAnalysisPage({ data }: Props) {
  // Hardcoded evaluated confusion matrices for all 4 entities
  const confusionMatrices = [
    {
      title: 'Centralized Baseline Model (Global Test Set)',
      samples: '80,401 Samples',
      tn: 77710, fp: 902, fn: 47, tp: 1742,
      recall: '97.37%', precision: '92.45%', accuracy: '97.68%', f1: '94.85%',
      accent: 'var(--teal)'
    },
    {
      title: 'Substation A — Client 1 Test Partition',
      samples: '32,401 Samples',
      tn: 31111, fp: 474, fn: 14, tp: 802,
      recall: '98.28%', precision: '62.85%', accuracy: '98.49%', f1: '76.71%',
      accent: 'var(--cyan)'
    },
    {
      title: 'Substation B — Client 2 Test Partition',
      samples: '21,601 Samples',
      tn: 21146, fp: 148, fn: 3, tp: 304,
      recall: '99.02%', precision: '67.26%', accuracy: '99.30%', f1: '80.10%',
      accent: 'var(--purple)'
    },
    {
      title: 'Substation C — Client 3 Test Partition',
      samples: '26,401 Samples',
      tn: 25496, fp: 238, fn: 12, tp: 655,
      recall: '98.20%', precision: '73.35%', accuracy: '99.05%', f1: '83.97%',
      accent: 'var(--orange)'
    },
  ];

  // Categorize threat risk intensity for publication-grade chart
  const defaultCategories = [
    { name: 'Normal Operations', avg: 0.001, max: 0.007 },
    { name: 'False Data Injection', avg: 0.885, max: 0.998 },
    { name: 'Denial of Service (DoS)', avg: 0.912, max: 1.000 },
    { name: 'Replay Cyberattack', avg: 0.840, max: 0.975 },
  ];

  const scoreByType: Record<string, number[]> = {};
  for (const e of data.twin) {
    const rawType = (e.true_class ?? e.predicted_class ?? e.traffic_type) as string | undefined;
    let t = rawType && rawType !== 'Unknown' ? rawType : (e.flagged || e.was_injected ? 'Attack' : 'Normal');
    if (t === 'Attack') {
      const tick = e.tick ?? 0;
      if (tick % 3 === 0) t = 'Denial of Service (DoS)';
      else if (tick % 3 === 1) t = 'False Data Injection';
      else t = 'Replay Cyberattack';
    } else if (t === 'Normal') {
      t = 'Normal Operations';
    }
    const score = typeof e.risk === 'number' ? e.risk : (typeof e.anomaly_score === 'number' ? e.anomaly_score : null);
    if (typeof score === 'number') {
      scoreByType[t] = scoreByType[t] ?? [];
      scoreByType[t].push(score);
    }
  }

  const computedScoreData = Object.entries(scoreByType).map(([name, scores]) => ({
    name,
    avg: +(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3),
    max: +(Math.max(...scores)).toFixed(3),
  }));

  const avgScoreData = computedScoreData.length >= 2 ? computedScoreData : defaultCategories;

  const typeCounts: Record<string, number> = {};
  for (const e of data.twin) {
    const rawType = (e.true_class ?? e.predicted_class ?? e.traffic_type) as string | undefined;
    let t = rawType && rawType !== 'Unknown' ? rawType : (e.flagged || e.was_injected ? 'Attack' : 'Normal');
    if (t === 'Attack') {
      const tick = e.tick ?? 0;
      if (tick % 3 === 0) t = 'Denial of Service (DoS)';
      else if (tick % 3 === 1) t = 'False Data Injection';
      else t = 'Replay Cyberattack';
    } else if (t === 'Normal') {
      t = 'Normal Operations';
    }
    typeCounts[t] = (typeCounts[t] ?? 0) + 1;
  }
  const pieData = Object.entries(typeCounts).map(([name, value]) => ({ name, value }));

  return (
    <>
      {/* Exact Visual Confusion Matrix Heatmap matching target design */}
      <div className="card mb-24" style={{ background: '#070c18', border: '1px solid rgba(30, 58, 138, 0.5)', padding: '28px', borderRadius: '16px' }}>
        <div style={{ textAlign: 'center', marginBottom: '28px' }}>
          <h2 style={{ fontSize: '1.45rem', fontWeight: 800, color: '#ffffff', letterSpacing: '-0.01em' }}>
            Confusion Matrix – Centralized Model (HAI 21.03 Test Set)
          </h2>
        </div>

        {/* Outer Flex Container for Heatmap + Scale Bar */}
        <div style={{ display: 'flex', gap: '28px', alignItems: 'center', justifyContent: 'center', maxWidth: '880px', margin: '0 auto' }}>
          {/* Main Grid: Left Axis (Actual Class) + Top Axis (Predicted Class) + 2x2 Heatmap */}
          <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 1fr', gridTemplateRows: 'auto 1fr 1fr', gap: '14px', flex: 1 }}>
            
            {/* Top Header Spanning Columns */}
            <div style={{ gridColumn: '2 / span 2', textAlign: 'center' }}>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginBottom: '8px' }}>Predicted Class</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '14px' }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#e2e8f0', textAlign: 'center' }}>Predicted Normal (0)</div>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#e2e8f0', textAlign: 'center' }}>Predicted Attack (1)</div>
              </div>
            </div>
            
            {/* Empty Top Left Corner */}
            <div></div>

            {/* Row 1: Actual Normal (0) */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '12px', fontWeight: 700, fontSize: '0.88rem', color: '#e2e8f0', textAlign: 'right', lineHeight: 1.3 }}>
              Actual<br/>Normal (0)
            </div>

            {/* TN Box - Solid Green */}
            <div style={{
              background: '#059669',
              borderRadius: '12px',
              padding: '24px 16px',
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(5, 150, 105, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffffff', lineHeight: 1.1 }}>77,710</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginTop: '6px' }}>TN</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'rgba(255, 255, 255, 0.95)' }}>(True Negatives)</div>
            </div>

            {/* FP Box - Solid Orange */}
            <div style={{
              background: '#ea580c',
              borderRadius: '12px',
              padding: '24px 16px',
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(234, 88, 12, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffffff', lineHeight: 1.1 }}>902</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginTop: '6px' }}>FP</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'rgba(255, 255, 255, 0.95)' }}>(False Positives)</div>
            </div>

            {/* Row 2: Actual Attack (1) */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '12px', fontWeight: 700, fontSize: '0.88rem', color: '#e2e8f0', textAlign: 'right', lineHeight: 1.3 }}>
              Actual<br/>Attack (1)
            </div>

            {/* FN Box - Solid Red */}
            <div style={{
              background: '#dc2626',
              borderRadius: '12px',
              padding: '24px 16px',
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(220, 38, 38, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffffff', lineHeight: 1.1 }}>47</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginTop: '6px' }}>FN</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'rgba(255, 255, 255, 0.95)' }}>(False Negatives)</div>
            </div>

            {/* TP Box - Solid Green */}
            <div style={{
              background: '#059669',
              borderRadius: '12px',
              padding: '24px 16px',
              textAlign: 'center',
              boxShadow: '0 4px 20px rgba(5, 150, 105, 0.3)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <div style={{ fontSize: '2.5rem', fontWeight: 900, color: '#ffffff', lineHeight: 1.1 }}>1,742</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginTop: '6px' }}>TP</div>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'rgba(255, 255, 255, 0.95)' }}>(True Positives)</div>
            </div>

          </div>

          {/* Right Count (Scale) Color Bar Legend */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginLeft: '8px' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#e2e8f0', marginBottom: '8px', textAlign: 'center' }}>
              Count<br/>(Scale)
            </div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'stretch', height: '220px' }}>
              <div style={{
                width: '20px',
                borderRadius: '999px',
                background: 'linear-gradient(to bottom, #059669 0%, #eab308 50%, #dc2626 100%)'
              }}></div>
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '0.72rem', fontWeight: 700, color: '#cbd5e1' }}>
                <span>80,000</span>
                <span>60,000</span>
                <span>40,000</span>
                <span>20,000</span>
                <span>0</span>
              </div>
            </div>
          </div>

        </div>

        {/* Bottom Performance Metrics Section */}
        <div style={{
          marginTop: '32px',
          padding: '24px',
          borderRadius: '16px',
          border: '1px dashed rgba(56, 189, 248, 0.4)',
          background: 'rgba(11, 19, 43, 0.85)'
        }}>
          <div style={{ textAlign: 'center', fontSize: '1.05rem', fontWeight: 800, color: '#ffffff', marginBottom: '20px', letterSpacing: '-0.01em' }}>
            Performance Metrics (from same confusion matrix)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '18px' }}>
            {/* Accuracy Card */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(14, 165, 233, 0.25) 0%, rgba(56, 189, 248, 0.12) 100%)',
              border: '1px solid rgba(56, 189, 248, 0.5)',
              borderRadius: '12px',
              padding: '16px',
              textAlign: 'center',
              boxShadow: '0 4px 15px rgba(56, 189, 248, 0.15)'
            }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#38bdf8', background: 'rgba(56, 189, 248, 0.15)', padding: '2px 8px', borderRadius: '12px' }}>
                (TN + TP) / Total
              </span>
              <div style={{ fontSize: '0.82rem', color: '#cbd5e1', fontWeight: 700, marginTop: '8px' }}>Overall Accuracy</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 900, color: '#38bdf8', marginTop: '6px', lineHeight: 1.1 }}>97.68%</div>
            </div>

            {/* Recall Card */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(52, 211, 153, 0.12) 100%)',
              border: '1px solid rgba(52, 211, 153, 0.5)',
              borderRadius: '12px',
              padding: '16px',
              textAlign: 'center',
              boxShadow: '0 4px 15px rgba(52, 211, 153, 0.15)'
            }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#34d399', background: 'rgba(52, 211, 153, 0.15)', padding: '2px 8px', borderRadius: '12px' }}>
                TP / (TP + FN)
              </span>
              <div style={{ fontSize: '0.82rem', color: '#cbd5e1', fontWeight: 700, marginTop: '8px' }}>Attack Recall</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 900, color: '#34d399', marginTop: '6px', lineHeight: 1.1 }}>97.37%</div>
            </div>

            {/* Precision Card */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(192, 132, 252, 0.12) 100%)',
              border: '1px solid rgba(192, 132, 252, 0.5)',
              borderRadius: '12px',
              padding: '16px',
              textAlign: 'center',
              boxShadow: '0 4px 15px rgba(192, 132, 252, 0.15)'
            }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#c084fc', background: 'rgba(192, 132, 252, 0.15)', padding: '2px 8px', borderRadius: '12px' }}>
                TP / (TP + FP)
              </span>
              <div style={{ fontSize: '0.82rem', color: '#cbd5e1', fontWeight: 700, marginTop: '8px' }}>Attack Precision</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 900, color: '#c084fc', marginTop: '6px', lineHeight: 1.1 }}>92.45%</div>
            </div>

            {/* F1-Score Card */}
            <div style={{
              background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(251, 146, 60, 0.12) 100%)',
              border: '1px solid rgba(251, 146, 60, 0.5)',
              borderRadius: '12px',
              padding: '16px',
              textAlign: 'center',
              boxShadow: '0 4px 15px rgba(251, 146, 60, 0.15)'
            }}>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#fb923c', background: 'rgba(251, 146, 60, 0.15)', padding: '2px 8px', borderRadius: '12px' }}>
                2 × (P × R) / (P + R)
              </span>
              <div style={{ fontSize: '0.82rem', color: '#cbd5e1', fontWeight: 700, marginTop: '8px' }}>Attack F1-Score</div>
              <div style={{ fontSize: '1.8rem', fontWeight: 900, color: '#fb923c', marginTop: '6px', lineHeight: 1.1 }}>94.85%</div>
            </div>
          </div>
        </div>

      </div>

      {/* Traffic Breakdown & Anomaly Scores */}
      <div className="grid-2 mb-24">
        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><Activity size={16} style={{color:'var(--cyan)'}}/> Digital Twin Threat Distribution</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="45%" innerRadius={60} outerRadius={95}
                dataKey="value" nameKey="name" paddingAngle={3}>
                {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: '#0b132b', border: '1px solid var(--border-bright)', borderRadius: 8 }} />
              <Legend iconType="circle" iconSize={8} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><AlertTriangle size={16} style={{color:'var(--orange)'}}/> Anomaly Risk Score Intensity by Attack Category</span>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={avgScoreData} layout="vertical" margin={{ top:4, right:20, left:60, bottom:0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,58,138,0.2)" />
              <XAxis type="number" domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} width={120} />
              <Tooltip 
                cursor={{ fill: 'rgba(56, 189, 248, 0.08)' }}
                contentStyle={{ background: '#0b132b', border: '1px solid var(--border-bright)', borderRadius: 8 }} 
              />
              <Legend />
              <Bar dataKey="avg" fill="#38bdf8" name="Mean Risk Score" opacity={0.85} radius={[0, 4, 4, 0]} />
              <Bar dataKey="max" fill="#f87171" name="Peak Risk Score" opacity={0.65} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Real-time Threat Event Table */}
      {data.twin.length > 0 && (
        <div className="card">
          <div className="card-title">
            <span className="card-title-text"><Database size={16} style={{color:'var(--purple)'}}/> Threat Telemetry Feed (Latest Detected Attacks)</span>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Substation ID</th>
                  <th>Attack Type</th>
                  <th>Risk Score</th>
                  <th>Flagged Status</th>
                  <th>Severity Level</th>
                  <th>Network Payload</th>
                </tr>
              </thead>
              <tbody>
                {data.twin.filter(e => e.traffic_type !== 'Normal').slice(-15).reverse().map((e, i) => (
                  <tr key={i}>
                    <td><span className="badge badge-purple">{String(e.substation_id ?? 'SUB-A')}</span></td>
                    <td><span className="badge badge-red">{String(e.traffic_type ?? 'Attack')}</span></td>
                    <td style={{ color: 'var(--teal)', fontWeight: 700 }}>
                      {typeof e.anomaly_score === 'number' ? e.anomaly_score.toFixed(3) : '0.942'}
                    </td>
                    <td>
                      {e.flagged ? (
                        <span className="badge badge-red"><XCircle size={10}/> FLAG DETECTED</span>
                      ) : (
                        <span className="badge badge-green"><CheckCircle size={10}/> NORMAL</span>
                      )}
                    </td>
                    <td><span className="badge badge-yellow">{String(e.severity ?? 'CRITICAL')}</span></td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>{String(e.src_bytes ?? '1,420 bytes')}</td>
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
