import { useState, useEffect, useCallback } from 'react';
import { Activity, Zap, Shield, Radio, GitBranch, Cpu, RefreshCw, Copy, Check, FileText, X } from 'lucide-react';
import { api, FLMetrics, BaselineMetrics, TwinEvent, NodeStatus } from './api';
import OverviewPage from './components/OverviewPage';
import FLTrainingPage from './components/FLTrainingPage';
import AttackAnalysisPage from './components/AttackAnalysisPage';
import DigitalTwinPage from './components/DigitalTwinPage';
import NodeStatusPage from './components/NodeStatusPage';

type Page = 'Overview' | 'FL Training' | 'Attack Analysis' | 'Digital Twin' | 'Node Status';

const NAV: { id: Page; icon: React.ReactNode; label: string }[] = [
  { id: 'Overview',        icon: <Activity size={16}/>,   label: 'Overview'        },
  { id: 'FL Training',     icon: <GitBranch size={16}/>,  label: 'FL Training'     },
  { id: 'Attack Analysis', icon: <Shield size={16}/>,     label: 'Attack Analysis' },
  { id: 'Digital Twin',    icon: <Cpu size={16}/>,        label: 'Digital Twin'    },
  { id: 'Node Status',     icon: <Radio size={16}/>,      label: 'Node Status'     },
];

const PAGE_META: Record<Page, { title: string; sub: string }> = {
  'Overview':        { title: '⚡ SmartGrid Executive Command Center', sub: 'Privacy-Preserving Cyberattack Detection — Real-time HAI 21.03 testbed analytics' },
  'FL Training':     { title: '🔄 Federated Learning Telemetry',       sub: '10-Round FedAvg convergence, loss curves & non-IID client contributions' },
  'Attack Analysis': { title: '🚨 Attack Analysis & Confusion Matrix', sub: 'Multi-model confusion matrix heatmaps, per-class recall & precision' },
  'Digital Twin':    { title: '🔗 Digital Twin Anomaly Monitor',      sub: 'Simulated 30-tick physical process risk scores & Autonomous Response Agent actions' },
  'Node Status':     { title: '📡 Substation Grid Node Status',        sub: 'Real-time telemetry and hardware status across Substation A, B, and C' },
};

export interface AppData {
  fl:       FLMetrics | null;
  baseline: BaselineMetrics | null;
  twin:     TwinEvent[];
  nodes:    NodeStatus | null;
  lastUpdated: string;
  loading: boolean;
  apiError: boolean;
}

export default function App() {
  const [page, setPage]             = useState<Page>('Overview');
  const [showLatex, setShowLatex]   = useState<boolean>(false);
  const [copied, setCopied]         = useState<boolean>(false);
  const [data, setData]             = useState<AppData>({
    fl: null, baseline: null, twin: [], nodes: null,
    lastUpdated: '—', loading: true, apiError: false,
  });

  const fetchAll = useCallback(async () => {
    try {
      const [fl, baseline, twin, nodes] = await Promise.all([
        api.flMetrics(),
        api.baseline(),
        api.twinEvents(300),
        api.nodeStatus(),
      ]);
      setData({
        fl, baseline, twin, nodes,
        lastUpdated: new Date().toLocaleTimeString(),
        loading: false,
        apiError: false,
      });
    } catch {
      setData(prev => ({ ...prev, loading: false, apiError: true }));
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const id = setInterval(fetchAll, 3000);
    return () => clearInterval(id);
  }, [fetchAll]);

  const meta = PAGE_META[page];

  const latexCode = `\\begin{table}[h!]
\\centering
\\caption{Performance comparison of Centralized Baseline vs. Federated SmartGrid Sentinel on the HAI 21.03 Dataset.}
\\label{tab:smartgrid_results}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Model Architecture} & \\textbf{Accuracy} & \\textbf{Attack Recall} & \\textbf{Attack Precision} & \\textbf{F1-Score} \\\\
\\hline
Centralized Baseline & 97.68\\% & 97.37\\% & 88.50\\% & 92.72\\% \\\\
Federated Learning (Round 10) & 97.68\\% & 96.85\\% & 89.55\\% & 93.05\\% \\\\
\\hline
\\end{tabular}
\\end{table}`;

  const copyLatex = () => {
    navigator.clipboard.writeText(latexCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1><Zap size={20} /> SmartGrid Sentinel</h1>
          <p>Privacy-Preserving Cyberattack<br/>Detection &amp; Automated Mitigation</p>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-label">Control Panel</div>
          {NAV.map(n => (
            <div
              key={n.id}
              className={`nav-item ${page === n.id ? 'active' : ''}`}
              onClick={() => setPage(n.id)}
            >
              {n.icon} {n.label}
            </div>
          ))}
        </div>

        <div className="sidebar-info">
          <div className="info-row">
            <span className="pulse-dot" />
            <span style={{fontWeight:600, color:'var(--text-primary)'}}>FL FedAvg Server: Active</span>
          </div>
          <div className="info-row" style={{marginTop:4}}>
            <Shield size={12} style={{color:'var(--teal)'}}/>
            <span>HAI 21.03 Testbed Corpus</span>
          </div>
          <div className="info-row" style={{marginTop:4}}>
            <Radio size={12} style={{color:'var(--cyan)'}}/>
            <span>3 Substation Non-IID Nodes</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <div className="main">
        <header className="topbar">
          <div>
            <h2>{meta.title}</h2>
            <p>{meta.sub}</p>
          </div>
          <div className="topbar-right">
            <button className="btn-latex" onClick={() => setShowLatex(true)}>
              <FileText size={14} /> Export Paper LaTeX Table
            </button>

            {data.apiError ? (
              <span className="badge badge-red">⚠ FastAPI Offline</span>
            ) : (
              <span className="refresh-badge">
                <RefreshCw size={12} /> Live Telemetry · {data.lastUpdated}
              </span>
            )}
          </div>
        </header>

        <div className="page-content">
          {data.loading && !data.apiError && (
            <div className="banner banner-info">
              <span className="pulse-dot" /> Connecting to SmartGrid Sentinel Backend API…
            </div>
          )}
          {data.apiError && (
            <div className="banner banner-warn">
              ⚠ Cannot reach FastAPI backend on port 8008. Run:{' '}
              <code style={{fontFamily:'var(--font-mono)', background:'rgba(0,0,0,0.4)', padding:'2px 8px', borderRadius:4, color:'var(--teal)'}}>
                bash start_dashboard.sh
              </code>
            </div>
          )}

          {page === 'Overview'        && <OverviewPage data={data} />}
          {page === 'FL Training'     && <FLTrainingPage data={data} />}
          {page === 'Attack Analysis' && <AttackAnalysisPage data={data} />}
          {page === 'Digital Twin'    && <DigitalTwinPage data={data} />}
          {page === 'Node Status'     && <NodeStatusPage data={data} />}
        </div>
      </div>

      {/* Paper LaTeX Modal */}
      {showLatex && (
        <div className="modal-overlay" onClick={() => setShowLatex(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📄 LaTeX Table Code for Research Paper</h3>
              <button 
                onClick={() => setShowLatex(false)} 
                style={{background:'none', border:'none', color:'var(--text-muted)', cursor:'pointer'}}
              >
                <X size={18} />
              </button>
            </div>
            <p style={{fontSize:'0.8rem', color:'var(--text-secondary)', marginBottom:12}}>
              Copy this LaTeX code snippet directly into your Overleaf / IEEE paper draft:
            </p>
            <div className="code-block">{latexCode}</div>
            <div style={{display:'flex', justifyContent:'flex-end', marginTop:16, gap:12}}>
              <button 
                className="btn-latex" 
                onClick={copyLatex}
                style={{padding:'8px 18px', fontSize:'0.82rem'}}
              >
                {copied ? <Check size={14}/> : <Copy size={14}/>}
                {copied ? 'Copied to Clipboard!' : 'Copy LaTeX Code'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
