import React, { useState, useEffect, useRef } from 'react';
import { 
  Sliders, 
  Activity, 
  Cpu, 
  ShieldAlert, 
  Play, 
  RefreshCw, 
  Info,
  Maximize2,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const BACKEND_URL = 'http://127.0.0.1:8000';

interface DotConfig {
  name: string;
  x0: number;
  y0: number;
  radius: number;
}

interface GateConfig {
  name: string;
  x0: number;
  y0: number;
  type: string;
  lever_arm: number;
}

interface TwinConfig {
  num_dots: number;
  temperature: number;
  noise_amplitude: number;
  noise_exponent: number;
  gate_voltages: Record<string, number>;
  sensor_weights: number[];
  dots: DotConfig[];
  gates: GateConfig[];
}

interface QuantumState {
  energy_eV: number;
  density: number[][];
}

interface WavefunctionResponse {
  x: number[];
  y: number[];
  states: QuantumState[];
}

interface DiagnosticResponse {
  status: string;
  issues: string[];
  timestamp: number;
}

export default function App() {
  // Config state
  const [config, setConfig] = useState<TwinConfig | null>(null);
  const [activeTab, setActiveTab] = useState<'potential' | 'stability' | 'tuning'>('potential');
  
  // Real-time synchronization values (from slider controls)
  const [voltages, setVoltages] = useState<Record<string, number>>({});
  const [tempMK, setTempMK] = useState<number>(50); // 50 mK
  const [noiseAmp, setNoiseAmp] = useState<number>(6.0e-5);
  const [noiseExp, setNoiseExp] = useState<number>(1.0);
  
  // Physics engine visual results
  const [potentialData, setPotentialData] = useState<number[][] | null>(null);
  const [quantumStates, setQuantumStates] = useState<QuantumState[] | null>(null);
  const [activeStateIndex, setActiveStateIndex] = useState<number>(0);
  
  // Charge stability diagram scan parameters and results
  const [scanResult, setScanResult] = useState<{
    v1_range: number[];
    v2_range: number[];
    sensor: number[][];
    derivative: number[][];
  } | null>(null);
  const [scanLoading, setScanLoading] = useState<boolean>(false);
  const [scanType, setScanType] = useState<'sensor' | 'derivative'>('derivative');
  
  // AI Tuning parameters
  const [tuningTarget, setTuningTarget] = useState<[number, number]>([1, 1]);
  const [tuningRunning, setTuningRunning] = useState<boolean>(false);
  const [tuningLog, setTuningLog] = useState<string[]>([]);
  const [tuningTrajectory, setTuningTrajectory] = useState<number[][]>([]);
  
  // Diagnostics
  const [diagnostics, setDiagnostics] = useState<DiagnosticResponse | null>(null);
  
  // Real-time synchronization outputs
  const [twinState, setTwinState] = useState<any>(null);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [isTwinActive, setIsTwinActive] = useState<boolean>(true);
  
  // Interactive 3D potential canvas rotation parameters
  const [pitch, setPitch] = useState<number>(0.6); // pitch (radians)
  const [yaw, setYaw] = useState<number>(0.7);   // yaw (radians)
  const isDragging = useRef<boolean>(false);
  const dragStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  
  // Refs
  const potentialCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stabilityCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);
  const [systemLogs, setSystemLogs] = useState<string[]>([]);

  // Add a log entry
  const addLog = (message: string) => {
    const timeStr = new Date().toLocaleTimeString();
    setSystemLogs(prev => [...prev.slice(-30), `[${timeStr}] ${message}`]);
  };

  // 1. Initial Load Config
  useEffect(() => {
    fetch(`${BACKEND_URL}/api/config`)
      .then(res => res.json())
      .then((data: TwinConfig) => {
        setConfig(data);
        setVoltages(data.gate_voltages);
        setTempMK(Math.round(data.temperature * 1000));
        setNoiseAmp(data.noise_amplitude);
        setNoiseExp(data.noise_exponent);
        addLog(`System configured: ${data.num_dots}-dot array loaded successfully.`);
      })
      .catch(err => {
        console.error(err);
        addLog("Error: Failed to connect to FastAPI backend. Make sure the server is running on port 8000.");
      });
  }, []);

  // Update backend when config inputs change (throttled/debounced implicitly by button or slider release)
  const syncConfig = (updatedVoltages = voltages, tK = tempMK / 1000, nAmp = noiseAmp, nExp = noiseExp, nDots = config?.num_dots) => {
    fetch(`${BACKEND_URL}/api/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temperature: tK,
        noise_amplitude: nAmp,
        noise_exponent: nExp,
        gate_voltages: updatedVoltages,
        num_dots: nDots
      })
    })
      .then(res => res.json())
      .then((data: TwinConfig) => {
        setConfig(data);
        setVoltages(data.gate_voltages);
      })
      .catch(err => console.error("Sync config error:", err));
  };

  // 2. Fetch Potential and Solve Schrödinger when voltages or layout change
  useEffect(() => {
    if (!config) return;
    
    // Fetch potential landscape map
    fetch(`${BACKEND_URL}/api/potential`)
      .then(res => res.json())
      .then(data => {
        setPotentialData(data.V);
      });
      
    // Fetch Wavefunctions
    fetch(`${BACKEND_URL}/api/wavefunctions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_states: 4 })
    })
      .then(res => res.json())
      .then((data: WavefunctionResponse) => {
        setQuantumStates(data.states);
      })
      .catch(err => console.error("Schrödinger request failed:", err));
  }, [voltages, config?.num_dots]);

  // 3. Digital Twin Real-Time Loop (Pink noise & sensor reading sync)
  useEffect(() => {
    if (!isTwinActive) return;
    
    const interval = setInterval(() => {
      // Periodic step to synchronize virtual state & read fluctuating variables
      fetch(`${BACKEND_URL}/api/step`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
          setTwinState(data);
          
          // Format data for Recharts history
          const hist = data.history.time.map((t: number, i: number) => ({
            time: t.toFixed(2),
            noise: data.history.noise[i] * 1000,  // convert to mV
            sensor: data.history.sensor[i]
          }));
          setHistoryData(hist);
        })
        .catch(err => console.error("Step sync error:", err));
    }, 250);
    
    return () => clearInterval(interval);
  }, [isTwinActive, noiseAmp, noiseExp, tempMK]);

  // 4. Run Stability scan
  const triggerScan = () => {
    setScanLoading(true);
    addLog("Initiating 2D gate voltage sweep (stability scan)...");
    
    fetch(`${BACKEND_URL}/api/stability`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        v1_min: 0.0,
        v1_max: 0.12,
        v2_min: 0.0,
        v2_max: 0.12,
        steps: 64,
        gate1: "P1",
        gate2: "P2",
        temperature: tempMK / 1000
      })
    })
      .then(res => res.json())
      .then(data => {
        setScanResult(data);
        setScanLoading(false);
        addLog("Charge stability scan complete. Honeycomb diagram rendered.");
      })
      .catch(err => {
        console.error(err);
        setScanLoading(false);
        addLog("Scan failed. Check backend connections.");
      });
  };

  // Run initial scan once config is loaded
  useEffect(() => {
    if (config && config.num_dots >= 2) {
      triggerScan();
    }
  }, [config?.num_dots]);

  // 5. Run diagnostics
  const runDiagnostics = () => {
    addLog("Running array lithography diagnostics and lever-arm calibration...");
    fetch(`${BACKEND_URL}/api/diagnose`)
      .then(res => res.json())
      .then((data: DiagnosticResponse) => {
        setDiagnostics(data);
        if (data.status === "HEALTHY") {
          addLog("Device diagnostic check passed. Status: healthy.");
        } else {
          addLog(`Device warning: ${data.issues.length} anomalies detected.`);
        }
      });
  };

  // 6. Run AI Auto-tuning closed-loop
  const triggerAutotune = () => {
    if (tuningRunning) return;
    setTuningRunning(true);
    setTuningLog(["[AI Solver] Initializing reinforcement learning agent..."]);
    addLog(`AI solver started: tuning plungers to reach target state (${tuningTarget[0]}, ${tuningTarget[1]})`);
    
    fetch(`${BACKEND_URL}/api/autotune`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        target_state_x: tuningTarget[0],
        target_state_y: tuningTarget[1],
        max_steps: 25,
        step_voltage: 3.5e-3
      })
    })
      .then(res => res.json())
      .then(data => {
        setTuningRunning(false);
        if (data.success) {
          addLog(`AI Success: target state reached in ${data.steps} steps.`);
          setTuningLog(prev => [
            ...prev,
            `[AI Solver] Target state achieved in ${data.steps} steps.`,
            `[AI Solver] Final voltages: P1 = ${data.final_voltages.P1.toFixed(4)}V, P2 = ${data.final_voltages.P2.toFixed(4)}V`
          ]);
        } else {
          addLog(`AI Alert: target state not found within limits.`);
          setTuningLog(prev => [...prev, "[AI Solver] Failed to reach target state. Step limit exceeded."]);
        }
        
        // Save trajectory
        setTuningTrajectory(data.voltage_trajectory);
        
        // Animate the sliders to mimic closed loop hardware tuning!
        let idx = 0;
        const animate = setInterval(() => {
          if (idx >= data.voltage_trajectory.length) {
            clearInterval(animate);
            // Final sync
            setVoltages({
              ...voltages,
              "P1": data.final_voltages.P1,
              "P2": data.final_voltages.P2
            });
            syncConfig({
              ...voltages,
              "P1": data.final_voltages.P1,
              "P2": data.final_voltages.P2
            });
          } else {
            const [v1, v2] = data.voltage_trajectory[idx];
            setVoltages(prev => ({ ...prev, "P1": v1, "P2": v2 }));
            idx++;
          }
        }, 100);
      })
      .catch(err => {
        console.error(err);
        setTuningRunning(false);
        addLog("Tuning fail. Server connectivity error.");
      });
  };

  // Scroll terminal logs automatically
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [systemLogs, tuningLog]);

  // 7. Interactive canvas painting: 3D potential mesh & wavefunction density
  useEffect(() => {
    const canvas = potentialCanvasRef.current;
    if (!canvas || !potentialData) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const Nx = potentialData.length;
    const Ny = potentialData[0].length;
    
    const cx = canvas.width / 2;
    const cy = canvas.height / 2 + 30;
    const scale = 250; // horizontal size
    const zScale = 1.8e3; // vertical scaling (eV to pixels)
    
    // Rotation matrices
    const cosRx = Math.cos(pitch);
    const sinRx = Math.sin(pitch);
    const cosRy = Math.cos(yaw);
    const sinRy = Math.sin(yaw);
    
    const project = (i: number, j: number, zVal: number) => {
      // Normalize to [-0.5, 0.5]
      const px = (i - Nx / 2) / Nx;
      const py = (j - Ny / 2) / Ny;
      
      // Rotate around Z axis (Yaw)
      const x1 = px * cosRy - py * sinRy;
      const y1 = px * sinRy + py * cosRy;
      
      // Rotate around X axis (Pitch)
      const x2 = x1;
      const y2 = y1 * cosRx - (zVal * 0.1) * sinRx;
      
      // 3D potential coordinates
      const zOffset = zVal * zScale;
      
      return {
        u: cx + x2 * scale,
        v: cy - y2 * scale * 0.6 - zOffset
      };
    };
    
    // Draw surface mesh lines
    ctx.lineWidth = 1.0;
    
    // Draw columns (y-slices)
    for (let i = 0; i < Nx; i += 2) {
      ctx.beginPath();
      for (let j = 0; j < Ny; j++) {
        const potential = potentialData[i][j]; // potential energy (eV)
        const pt = project(i, j, potential);
        
        if (j === 0) ctx.moveTo(pt.u, pt.v);
        else ctx.lineTo(pt.u, pt.v);
      }
      
      // Draw grid line coloring
      // Highlight lines containing wavefunction probability density
      const densitySlice = quantumStates?.[activeStateIndex]?.density?.[i];
      const maxDensity = densitySlice ? Math.max(...densitySlice) : 0.0;
      
      if (maxDensity > 0.08) {
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.55)'; // glowing indigo
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)'; // dark slate
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
    }
    
    // Draw rows (x-slices)
    for (let j = 0; j < Ny; j += 2) {
      ctx.beginPath();
      for (let i = 0; i < Nx; i++) {
        const potential = potentialData[i][j];
        const pt = project(i, j, potential);
        
        if (i === 0) ctx.moveTo(pt.u, pt.v);
        else ctx.lineTo(pt.u, pt.v);
      }
      
      // Determine if active wavefunction spans this slice
      let maxDensity = 0;
      if (quantumStates?.[activeStateIndex]?.density) {
        for (let i = 0; i < Nx; i++) {
          const dens = quantumStates[activeStateIndex].density[i][j];
          if (dens > maxDensity) maxDensity = dens;
        }
      }
      
      if (maxDensity > 0.08) {
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.55)'; // glowing rose
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = 'rgba(51, 65, 85, 0.25)';
        ctx.lineWidth = 0.8;
      }
      ctx.stroke();
    }
    
    // Annotate quantum dots positions
    if (config) {
      ctx.fillStyle = '#10b981';
      config.dots.forEach(dot => {
        // Convert physical position (meters) to grid coordinate index
        const extentX = config.gates[0]?.lever_arm ? 120e-9 : 120e-9;
        const i_grid = Math.round((dot.x0 + extentX/2) / extentX * Nx);
        const j_grid = Math.round((dot.y0 + extentX/2) / extentX * Ny);
        
        if (i_grid >= 0 && i_grid < Nx && j_grid >= 0 && j_grid < Ny) {
          const potential = potentialData[i_grid][j_grid];
          const pt = project(i_grid, j_grid, potential);
          ctx.beginPath();
          ctx.arc(pt.u, pt.v, 4, 0, 2*Math.PI);
          ctx.fill();
          ctx.fillStyle = '#f8fafc';
          ctx.font = '10px JetBrains Mono';
          ctx.fillText(dot.name.toUpperCase(), pt.u - 8, pt.v - 8);
          ctx.fillStyle = '#10b981';
        }
      });
    }
    
  }, [potentialData, quantumStates, activeStateIndex, pitch, yaw, config]);

  // Drag interaction to rotate potential landscape
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    dragStart.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return;
    const dx = e.clientX - dragStart.current.x;
    const dy = e.clientY - dragStart.current.y;
    
    setYaw(prev => prev + dx * 0.007);
    setPitch(prev => Math.max(0.1, Math.min(Math.PI / 2 - 0.05, prev + dy * 0.007)));
    
    dragStart.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseUp = () => {
    isDragging.current = false;
  };

  // 8. Paint Charge stability diagram (2D heatmap on Canvas)
  useEffect(() => {
    const canvas = stabilityCanvasRef.current;
    if (!canvas || !scanResult) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const grid = scanType === 'sensor' ? scanResult.sensor : scanResult.derivative;
    const Ny = grid.length;
    const Nx = grid[0].length;
    
    const cw = canvas.width;
    const ch = canvas.height;
    
    const cellW = cw / Nx;
    const cellH = ch / Ny;
    
    // Find min and max values for scaling
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let i = 0; i < Ny; i++) {
      for (let j = 0; j < Nx; j++) {
        const val = grid[i][j];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    
    const range = maxVal - minVal || 1.0;
    
    // Render pixels
    for (let i = 0; i < Ny; i++) {
      for (let j = 0; j < Nx; j++) {
        const val = grid[i][j];
        const norm = (val - minVal) / range;
        
        // High-tech dark cyan/blue map for sensor, and glowing red/pink for transitions (derivative)
        let r, g, b;
        if (scanType === 'sensor') {
          // Dark space theme: deep blue to cyan
          r = Math.round(3 + norm * 40);
          g = Math.round(15 + norm * 150);
          b = Math.round(35 + norm * 220);
        } else {
          // Derivative: black/blue boundaries to glowing rose pink lines
          r = Math.round(15 + Math.pow(norm, 2) * 230);
          g = Math.round(20 + Math.pow(norm, 2) * 40);
          b = Math.round(45 + Math.pow(norm, 2) * 120);
        }
        
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        // Standard grid is (row, col) => (i, j). We draw (x, y) = (j, Ny-1-i) to flip vertical axis
        ctx.fillRect(j * cellW, (Ny - 1 - i) * cellH, cellW + 1, cellH + 1);
      }
    }
    
    // Draw current voltages crosshair
    const v1_min = scanResult.v1_range[0];
    const v1_max = scanResult.v1_range[scanResult.v1_range.length - 1];
    const v2_min = scanResult.v2_range[0];
    const v2_max = scanResult.v2_range[scanResult.v2_range.length - 1];
    
    const currentV1 = voltages["P1"] || 0.0;
    const currentV2 = voltages["P2"] || 0.0;
    
    const ux = ((currentV1 - v1_min) / (v1_max - v1_min)) * cw;
    const uy = ch - ((currentV2 - v2_min) / (v2_max - v2_min)) * ch;
    
    // Crosshair circle
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(ux, uy, 6, 0, 2*Math.PI);
    ctx.stroke();
    
    // Dotted crosshair lines
    ctx.setLineDash([2, 4]);
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
    
    ctx.beginPath();
    ctx.moveTo(ux, 0);
    ctx.lineTo(ux, ch);
    ctx.moveTo(0, uy);
    ctx.lineTo(cw, uy);
    ctx.stroke();
    ctx.setLineDash([]); // reset
    
  }, [scanResult, voltages, scanType]);

  // Click on stability diagram to set voltages
  const handleStabilityClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = stabilityCanvasRef.current;
    if (!canvas || !scanResult) return;
    
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    const v1_min = scanResult.v1_range[0];
    const v1_max = scanResult.v1_range[scanResult.v1_range.length - 1];
    const v2_min = scanResult.v2_range[0];
    const v2_max = scanResult.v2_range[scanResult.v2_range.length - 1];
    
    const newV1 = v1_min + (clickX / canvas.width) * (v1_max - v1_min);
    const newV2 = v2_max - (clickY / canvas.height) * (v2_max - v2_min);
    
    const nextVoltages = {
      ...voltages,
      "P1": Math.max(v1_min, Math.min(v1_max, newV1)),
      "P2": Math.max(v2_min, Math.min(v2_max, newV2))
    };
    
    setVoltages(nextVoltages);
    syncConfig(nextVoltages);
    addLog(`Voltages updated from diagram click: P1 = ${newV1.toFixed(3)}V, P2 = ${newV2.toFixed(3)}V`);
  };

  const resetVoltages = () => {
    const nextVoltages = { ...voltages };
    Object.keys(nextVoltages).forEach(k => {
      nextVoltages[k] = k.startsWith("B") ? 0.2 : 0.0;
    });
    setVoltages(nextVoltages);
    syncConfig(nextVoltages);
    addLog("Device gate voltages reset to baseline (plungers at 0V).");
  };

  return (
    <div className="min-h-screen bg-quantum-dark text-quantum-text flex flex-col antialiased selection:bg-indigo-500 selection:text-white">
      {/* Header Bar */}
      <header className="border-b border-quantum-border bg-slate-950/80 px-6 py-4 flex items-center justify-between sticky top-0 z-40 backdrop-blur-md">
        <div className="flex items-center space-x-3">
          <div className="bg-indigo-600 text-white p-2 rounded-lg shadow-lg shadow-indigo-600/20">
            <Activity className="h-6 w-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-indigo-400 via-purple-400 to-rose-400 bg-clip-text text-transparent">
              QuantumTwin
            </h1>
            <p className="text-xs text-quantum-muted font-medium uppercase tracking-wider">
              AI-Assisted Silicon Qubit Digital Twin
            </p>
          </div>
        </div>
        
        {/* Connection status and config state */}
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center space-x-2 bg-slate-900 border border-quantum-border py-1.5 px-3 rounded-full">
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-500 animate-ping"></span>
            <span className="text-xs font-mono font-semibold text-emerald-400 uppercase tracking-wider">
              ONLINE
            </span>
          </div>
          <button 
            onClick={() => syncConfig()}
            className="flex items-center space-x-1.5 text-xs text-quantum-muted hover:text-indigo-400 border border-quantum-border bg-slate-900 py-1.5 px-3 rounded-full hover:border-indigo-500 transition-all active:scale-95"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>SYNC PARAMETERS</span>
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 p-6 grid grid-cols-12 gap-6 max-w-[1700px] mx-auto w-full">
        
        {/* Column 1: Controls Panel (Left Panel - Span 3) */}
        <div className="col-span-12 lg:col-span-3 flex flex-col space-y-6">
          
          {/* Controls Card */}
          <div className="glass-panel rounded-2xl p-5 shadow-xl flex flex-col space-y-5">
            <div className="flex items-center space-x-2 border-b border-quantum-border pb-3">
              <Sliders className="h-5 w-5 text-indigo-400" />
              <h2 className="font-semibold text-base uppercase tracking-wider">Device Architecture</h2>
            </div>
            
            {/* Array Selector */}
            <div className="flex flex-col space-y-2">
              <label className="text-xs font-semibold text-quantum-muted uppercase">Confinement Configuration</label>
              <div className="grid grid-cols-3 gap-2">
                {[1, 2, 3].map(num => (
                  <button
                    key={num}
                    onClick={() => {
                      if (config) {
                        syncConfig(voltages, tempMK/1000, noiseAmp, noiseExp, num);
                        addLog(`Layout reconfiguration: initialized ${num}-dot array.`);
                      }
                    }}
                    className={`py-2 text-xs font-mono font-semibold border rounded-lg transition-all ${
                      config?.num_dots === num 
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 shadow-md shadow-indigo-600/10' 
                        : 'bg-slate-900/60 border-quantum-border hover:border-slate-600 text-quantum-muted'
                    }`}
                  >
                    {num === 1 ? '1 DOT' : num === 2 ? 'DOUBLE' : 'TRIPLE'}
                  </button>
                ))}
              </div>
            </div>

            {/* Slider Voltages */}
            <div className="flex flex-col space-y-4 pt-2">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-quantum-muted uppercase">Gate Voltages</label>
                <button onClick={resetVoltages} className="text-[10px] font-semibold text-rose-400 hover:underline">RESET ALL</button>
              </div>
              
              {Object.keys(voltages).map(gateName => {
                const isBarrier = gateName.startsWith("B");
                const maxVal = isBarrier ? 0.5 : 0.15;
                const minVal = isBarrier ? 0.0 : 0.0;
                
                return (
                  <div key={gateName} className="flex flex-col space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-mono font-bold text-slate-300">{gateName} Plunger</span>
                      <span className={`font-mono ${isBarrier ? 'text-indigo-400' : 'text-emerald-400'}`}>
                        {voltages[gateName]?.toFixed(4)} V
                      </span>
                    </div>
                    <input
                      type="range"
                      min={minVal}
                      max={maxVal}
                      step="0.0005"
                      value={voltages[gateName] || 0.0}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        const nextVoltages = { ...voltages, [gateName]: val };
                        setVoltages(nextVoltages);
                      }}
                      onMouseUp={() => syncConfig()}
                      className="w-full h-1 bg-slate-900 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                  </div>
                );
              })}
            </div>

            {/* Physical parameters */}
            <div className="flex flex-col space-y-4 pt-4 border-t border-quantum-border/60">
              <label className="text-xs font-semibold text-quantum-muted uppercase">Environment & Noise</label>
              
              {/* Temperature */}
              <div className="flex flex-col space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-mono text-slate-300">Temperature</span>
                  <span className="font-mono text-indigo-400">{tempMK} mK</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="1200"
                  step="10"
                  value={tempMK}
                  onChange={(e) => {
                    const val = parseInt(e.target.value);
                    setTempMK(val);
                  }}
                  onMouseUp={() => syncConfig(voltages, tempMK/1000)}
                  className="w-full h-1 bg-slate-900 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              {/* Noise Amplitude */}
              <div className="flex flex-col space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-mono text-slate-300">1/f Noise Amplitude</span>
                  <span className="font-mono text-rose-400">{(noiseAmp * 1e6).toFixed(1)} μV</span>
                </div>
                <input
                  type="range"
                  min="5e-6"
                  max="3e-4"
                  step="5e-6"
                  value={noiseAmp}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setNoiseAmp(val);
                  }}
                  onMouseUp={() => syncConfig(voltages, tempMK/1000, noiseAmp)}
                  className="w-full h-1 bg-slate-900 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              {/* Real-time sync activation */}
              <div className="flex items-center justify-between pt-2">
                <span className="text-xs text-slate-300">Continuous Sync Loop</span>
                <button
                  onClick={() => setIsTwinActive(!isTwinActive)}
                  className={`w-12 h-6 flex items-center rounded-full p-0.5 transition-all ${
                    isTwinActive ? 'bg-indigo-600 justify-end' : 'bg-slate-800 justify-start'
                  }`}
                >
                  <span className="bg-white w-5 h-5 rounded-full shadow-md"></span>
                </button>
              </div>
            </div>

            {/* Diagnostic trigger */}
            <button
              onClick={runDiagnostics}
              className="mt-2 w-full py-2.5 bg-slate-900 hover:bg-slate-800/80 border border-quantum-border/80 rounded-xl text-xs font-semibold text-slate-200 flex items-center justify-center space-x-1.5 transition-all hover:border-slate-500 active:scale-95"
            >
              <ShieldAlert className="h-4 w-4 text-rose-400" />
              <span>RUN DIAGNOSTIC CHECKS</span>
            </button>
          </div>
        </div>

        {/* Column 2: Dashboard Visualization Panel (Center/Right Panel - Span 9) */}
        <div className="col-span-12 lg:col-span-9 flex flex-col space-y-6">
          
          {/* Main Visualizations Tabs card */}
          <div className="glass-panel rounded-2xl p-5 shadow-xl flex-1 flex flex-col min-h-[500px]">
            
            {/* Tabs Header */}
            <div className="flex items-center justify-between border-b border-quantum-border pb-3 mb-5">
              <div className="flex space-x-3">
                <button
                  onClick={() => setActiveTab('potential')}
                  className={`px-4 py-2 text-xs font-bold rounded-lg transition-all ${
                    activeTab === 'potential'
                      ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 shadow-sm'
                      : 'text-quantum-muted hover:text-slate-300'
                  }`}
                >
                  POTENTIAL & QUANTUM STATES
                </button>
                {config && config.num_dots >= 2 && (
                  <button
                    onClick={() => setActiveTab('stability')}
                    className={`px-4 py-2 text-xs font-bold rounded-lg transition-all ${
                      activeTab === 'stability'
                        ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 shadow-sm'
                        : 'text-quantum-muted hover:text-slate-300'
                    }`}
                  >
                    CHARGE STABILITY DIAGRAM
                  </button>
                )}
                <button
                  onClick={() => setActiveTab('tuning')}
                  className={`px-4 py-2 text-xs font-bold rounded-lg transition-all ${
                    activeTab === 'tuning'
                      ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 shadow-sm'
                      : 'text-quantum-muted hover:text-slate-300'
                  }`}
                >
                  AI AUTO-TUNING
                </button>
              </div>
              
              <div className="text-xs text-quantum-muted font-mono hidden md:block">
                Confinement: <span className="text-indigo-400 font-bold">{config?.num_dots || 2} QD Array</span>
              </div>
            </div>

            {/* TAB CONTENT: 3D Potential & wavefunctions */}
            {activeTab === 'potential' && (
              <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6">
                
                {/* 3D Canvas rendering (Span 8) */}
                <div className="md:col-span-8 flex flex-col justify-center items-center relative bg-slate-950/60 border border-quantum-border/60 rounded-xl p-4 min-h-[360px]">
                  <div className="absolute top-3 left-3 text-xs bg-slate-900/80 border border-quantum-border/60 py-1.5 px-3 rounded-md flex items-center space-x-1.5">
                    <Info className="h-3.5 w-3.5 text-indigo-400" />
                    <span className="text-slate-400">Drag mouse to rotate 3D view</span>
                  </div>
                  
                  <canvas
                    ref={potentialCanvasRef}
                    width={560}
                    height={380}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    className="cursor-grab active:cursor-grabbing max-w-full"
                  />
                  
                  <div className="absolute bottom-3 right-3 text-xs font-mono text-quantum-muted">
                    Pitch: {pitch.toFixed(2)} rad | Yaw: {yaw.toFixed(2)} rad
                  </div>
                </div>

                {/* Energy Level Spectrum (Span 4) */}
                <div className="md:col-span-4 flex flex-col space-y-4">
                  <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">Solved Energy Spectrum</h3>
                  
                  {/* Visual Spectrum representation */}
                  <div className="flex-1 bg-slate-950/40 border border-quantum-border/60 rounded-xl p-4 flex flex-col justify-between min-h-[300px]">
                    <div className="relative flex-1 flex flex-col justify-around py-4 border-l border-slate-700 ml-6 pl-4">
                      {quantumStates ? (
                        quantumStates.map((state, idx) => (
                          <div 
                            key={idx}
                            onClick={() => setActiveStateIndex(idx)}
                            className={`group cursor-pointer relative py-2 border-b border-dashed border-slate-700/60 hover:border-slate-500 transition-all ${
                              activeStateIndex === idx ? 'bg-indigo-600/5 -mx-4 px-4 border-slate-400' : ''
                            }`}
                          >
                            {/* Horizontal level bar */}
                            <div className={`h-1 rounded-full transition-all ${
                              activeStateIndex === idx 
                                ? 'bg-indigo-500 w-32 shadow-md shadow-indigo-600/40' 
                                : 'bg-slate-600 w-24 group-hover:bg-slate-400'
                            }`} />
                            
                            <div className="absolute -left-12 top-1.5 text-[10px] font-mono text-slate-400">
                              {(state.energy_eV * 1000).toFixed(2)}
                            </div>
                            
                            <div className="absolute left-32 top-1 flex items-center space-x-2 text-xs">
                              <span className={`font-mono ${activeStateIndex === idx ? 'text-indigo-400 font-bold' : 'text-quantum-muted'}`}>
                                {idx === 0 ? 'Ground state' : `Excited state ${idx}`}
                              </span>
                              {activeStateIndex === idx && <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-quantum-muted animate-pulse">Solving Hamiltonian...</div>
                      )}
                    </div>
                    
                    <div className="text-[10px] text-quantum-muted border-t border-quantum-border/60 pt-3 flex items-center space-x-1.5">
                      <Maximize2 className="h-3 w-3 text-indigo-400" />
                      <span>Energies are in meV. Click state to highlight wavefunction envelope.</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TAB CONTENT: Charge Stability diagram */}
            {activeTab === 'stability' && (
              <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6">
                
                {/* 2D scan diagram (Span 8) */}
                <div className="md:col-span-8 flex flex-col justify-center items-center relative bg-slate-950/60 border border-quantum-border/60 rounded-xl p-4 min-h-[360px]">
                  
                  {scanLoading ? (
                    <div className="flex flex-col items-center justify-center space-y-3">
                      <RefreshCw className="h-8 w-8 text-indigo-400 animate-spin" />
                      <p className="text-xs font-mono text-quantum-muted">Scanning gate voltages... solving charge states...</p>
                    </div>
                  ) : scanResult ? (
                    <div className="relative">
                      {/* Diagram Y Axis label (P2 Plunger) */}
                      <div className="absolute -left-10 top-1/2 -rotate-90 -translate-y-1/2 text-xs font-mono text-quantum-muted tracking-wider uppercase">
                        P2 Voltage (V)
                      </div>
                      
                      {/* Diagram X Axis label (P1 Plunger) */}
                      <div className="absolute bottom-[-24px] left-1/2 -translate-x-1/2 text-xs font-mono text-quantum-muted tracking-wider uppercase">
                        P1 Voltage (V)
                      </div>

                      <canvas
                        ref={stabilityCanvasRef}
                        width={360}
                        height={360}
                        onClick={handleStabilityClick}
                        className="border border-quantum-border rounded-lg cursor-crosshair shadow-2xl hover:border-slate-500 transition-all"
                      />
                    </div>
                  ) : (
                    <div className="text-xs text-quantum-muted">No scan data available. Trigger scan below.</div>
                  )}
                </div>

                {/* Scan Options (Span 4) */}
                <div className="md:col-span-4 flex flex-col space-y-4 justify-between">
                  <div className="flex flex-col space-y-4">
                    <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">Stability Configurations</h3>
                    
                    {/* Toggle Plot types */}
                    <div className="flex flex-col space-y-2">
                      <span className="text-[10px] font-semibold text-quantum-muted uppercase">Plot Type</span>
                      <div className="grid grid-cols-2 gap-2">
                        <button
                          onClick={() => setScanType('sensor')}
                          className={`py-2 text-xs font-semibold border rounded-lg transition-all ${
                            scanType === 'sensor' 
                              ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300' 
                              : 'bg-slate-900/60 border-quantum-border text-quantum-muted'
                          }`}
                        >
                          Raw Sensor Reading
                        </button>
                        <button
                          onClick={() => setScanType('derivative')}
                          className={`py-2 text-xs font-semibold border rounded-lg transition-all ${
                            scanType === 'derivative' 
                              ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300' 
                              : 'bg-slate-900/60 border-quantum-border text-quantum-muted'
                          }`}
                        >
                          Honeycomb Derivative
                        </button>
                      </div>
                    </div>
                    
                    <div className="bg-slate-950/40 border border-quantum-border/60 rounded-xl p-4 text-xs leading-relaxed space-y-2">
                      <div className="flex items-center space-x-2 text-indigo-400 font-semibold mb-1">
                        <Info className="h-4 w-4" />
                        <span>Honeycomb Boundaries</span>
                      </div>
                      <p className="text-slate-400">
                        The hexagonal honeycomb patterns outline the discrete charge states 
                        <strong> (N_1, N_2)</strong>.
                      </p>
                      <p className="text-slate-400">
                        Clicking inside the grid moves plunger gates directly to that voltage operating point in real-time.
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={triggerScan}
                    disabled={scanLoading}
                    className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center justify-center space-x-1.5 shadow-lg shadow-indigo-600/20 active:scale-95 disabled:opacity-50 transition-all"
                  >
                    <RefreshCw className={`h-4 w-4 ${scanLoading ? 'animate-spin' : ''}`} />
                    <span>REFRESH stability PLOT</span>
                  </button>
                </div>
              </div>
            )}

            {/* TAB CONTENT: AI Auto-tuning */}
            {activeTab === 'tuning' && (
              <div className="flex-1 grid grid-cols-1 md:grid-cols-12 gap-6">
                
                {/* Visual Trajectory graph & terminal logs (Span 8) */}
                <div className="md:col-span-8 flex flex-col space-y-4">
                  <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">AI Autonomous Solver Trajectory</h3>
                  
                  <div className="flex-1 bg-slate-950/60 border border-quantum-border/60 rounded-xl p-4 flex flex-col justify-between min-h-[300px]">
                    {tuningTrajectory.length > 0 ? (
                      <div className="flex-1 flex items-center justify-center p-2 relative h-[250px] w-full">
                        {/* Custom visual rendering of the search path */}
                        <div className="absolute top-2 left-2 text-[10px] font-mono text-quantum-muted">
                          Plunger Voltage Space (V_1, V_2)
                        </div>
                        
                        {/* Plotting path */}
                        <div className="h-full w-full flex items-center justify-center bg-slate-950/80 border border-slate-900 rounded-lg p-4 relative">
                          <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                            {/* Draw simple grid */}
                            <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
                            <line x1="50" y1="0" x2="50" y2="100" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
                            
                            {/* Target circle */}
                            <circle cx="70" cy="30" r="4" fill="rgba(16, 185, 129, 0.2)" stroke="#10b981" strokeWidth="1" />
                            <text x="73" y="32" fill="#10b981" fontSize="5" fontFamily="monospace">Target (1,1)</text>
                            
                            {/* Search path */}
                            {(() => {
                              const points = tuningTrajectory.map(pt => {
                                // Map V in [0.0, 0.12] to [10, 90] svg space
                                const sx = 10 + (pt[0] / 0.12) * 80;
                                const sy = 90 - (pt[1] / 0.12) * 80;
                                return `${sx},${sy}`;
                              }).join(' ');
                              
                              return (
                                <>
                                  <polyline points={points} fill="none" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="1,1" />
                                  {tuningTrajectory.map((pt, i) => {
                                    const sx = 10 + (pt[0] / 0.12) * 80;
                                    const sy = 90 - (pt[1] / 0.12) * 80;
                                    return (
                                      <circle 
                                        key={i} 
                                        cx={sx} 
                                        cy={sy} 
                                        r={i === 0 ? "2.5" : i === tuningTrajectory.length-1 ? "3.5" : "1.5"} 
                                        fill={i === 0 ? "#f43f5e" : i === tuningTrajectory.length-1 ? "#10b981" : "#6366f1"} 
                                      />
                                    );
                                  })}
                                </>
                              );
                            })()}
                          </svg>
                        </div>
                      </div>
                    ) : (
                      <div className="flex-1 flex items-center justify-center border border-dashed border-quantum-border/60 rounded-lg text-xs font-mono text-quantum-muted">
                        No tuning run recorded. Click "Auto-Tune" to start solver.
                      </div>
                    )}
                    
                    <div className="text-[10px] text-quantum-muted border-t border-quantum-border/60 pt-3">
                      The path shows plunger voltages stepping towards the single electron boundaries (glowing green).
                    </div>
                  </div>
                </div>

                {/* AI control logs (Span 4) */}
                <div className="md:col-span-4 flex flex-col space-y-4 justify-between">
                  <div className="flex flex-col space-y-4">
                    <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">AI Solver Setup</h3>
                    
                    {/* Setup inputs */}
                    <div className="flex flex-col space-y-3">
                      <div className="flex flex-col space-y-1.5">
                        <label className="text-[10px] font-semibold text-quantum-muted uppercase">Target Charge Configuration</label>
                        <select 
                          value={`${tuningTarget[0]},${tuningTarget[1]}`} 
                          onChange={(e) => {
                            const [x, y] = e.target.value.split(',').map(Number);
                            setTuningTarget([x, y]);
                          }}
                          className="bg-slate-900 border border-quantum-border py-2 px-3 rounded-lg text-xs font-mono font-bold text-slate-300 focus:outline-none focus:border-indigo-500"
                        >
                          <option value="1,1">Double Dot (1, 1) - Single Electron</option>
                          <option value="1,0">Left Dot (1, 0) - Charge degeneracy</option>
                          <option value="0,1">Right Dot (0, 1) - Charge degeneracy</option>
                          <option value="2,2">Double Dot (2, 2) - Two electron</option>
                        </select>
                      </div>

                      <div className="bg-slate-950/40 border border-quantum-border/60 rounded-xl p-4 text-xs space-y-2">
                        <div className="flex items-center space-x-1.5 text-emerald-400 font-bold">
                          <Cpu className="h-4 w-4" />
                          <span>DQN Tuning Solver</span>
                        </div>
                        <p className="text-slate-400 leading-relaxed">
                          Closed-loop tuning agent trained using Q-learning. Reads dot charge expectations and outputs plunger adjustments.
                        </p>
                      </div>
                    </div>
                  </div>

                  <button
                    onClick={triggerAutotune}
                    disabled={tuningRunning || config?.num_dots !== 2}
                    className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-bold flex items-center justify-center space-x-2 shadow-lg shadow-emerald-600/20 active:scale-95 disabled:opacity-50 transition-all"
                  >
                    {tuningRunning ? (
                      <RefreshCw className="h-4 w-4 animate-spin" />
                    ) : (
                      <Play className="h-4 w-4 fill-current" />
                    )}
                    <span>EXECUTE AUTOTUNE</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

      </main>

      {/* Bottom Panel: Analytics, Noise, Diagnostics, and Logs */}
      <footer className="border-t border-quantum-border bg-slate-950/60 p-6 grid grid-cols-12 gap-6 max-w-[1700px] mx-auto w-full mt-auto">
        
        {/* Real-time Noise streams (Span 4) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">Pink Noise Fluctuations</h3>
            {twinState && (
              <span className="text-[10px] font-mono text-rose-400">
                Gate Drift: {(twinState.noise_voltage * 1000).toFixed(3)} mV
              </span>
            )}
          </div>
          
          <div className="bg-slate-950/50 border border-quantum-border/60 rounded-xl p-4 h-[140px] flex items-center">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <XAxis dataKey="time" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip 
                  contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="noise" 
                  stroke="#f43f5e" 
                  strokeWidth={1.5} 
                  dot={false} 
                  name="Noise (mV)"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Real-time Sensor stream (Span 4) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">Charge Sensor Readout</h3>
            {twinState && (
              <span className="text-[10px] font-mono text-emerald-400">
                Conductance: {twinState.sensor_reading.toFixed(4)} G_0
              </span>
            )}
          </div>
          
          <div className="bg-slate-950/50 border border-quantum-border/60 rounded-xl p-4 h-[140px] flex items-center">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <XAxis dataKey="time" hide />
                <YAxis domain={['auto', 'auto']} hide />
                <Tooltip 
                  contentStyle={{ background: '#0f172a', border: '1px solid #334155' }}
                  labelStyle={{ color: '#94a3b8' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="sensor" 
                  stroke="#10b981" 
                  strokeWidth={1.5} 
                  dot={false} 
                  name="Sensor G_0"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Console / Diagnostics (Span 4) */}
        <div className="col-span-12 lg:col-span-4 flex flex-col space-y-3">
          <h3 className="text-xs font-bold text-quantum-muted uppercase tracking-wider">AI Diagnostics Console</h3>
          
          <div className="bg-slate-950/80 border border-quantum-border/60 rounded-xl p-4 h-[140px] overflow-y-auto flex flex-col space-y-2">
            
            {/* System health warning */}
            {diagnostics ? (
              <div className={`p-2 rounded-lg flex items-start space-x-2 border ${
                diagnostics.status === "HEALTHY" 
                  ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" 
                  : "bg-rose-500/10 border-rose-500/20 text-rose-400"
              }`}>
                {diagnostics.status === "HEALTHY" ? (
                  <CheckCircle2 className="h-4.5 w-4.5 shrink-0 mt-0.5" />
                ) : (
                  <AlertTriangle className="h-4.5 w-4.5 shrink-0 mt-0.5" />
                )}
                <div className="text-xs">
                  <div className="font-bold uppercase tracking-wider">Device {diagnostics.status}</div>
                  {diagnostics.issues.length > 0 ? (
                    <ul className="list-disc list-inside mt-1 font-mono text-[10px] space-y-0.5 text-slate-300">
                      {diagnostics.issues.map((iss, i) => <li key={i}>{iss}</li>)}
                    </ul>
                  ) : (
                    <p className="text-[10px] text-slate-300">All gates and dot dimensions verified within operating standards.</p>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-xs text-quantum-muted flex items-center space-x-2 py-2">
                <Info className="h-4 w-4 text-indigo-400" />
                <span>Run diagnostic checks to calibrate and diagnose array coupling.</span>
              </div>
            )}
            
            {/* Real-time system terminal messages */}
            <div className="flex-1 overflow-y-auto border-t border-quantum-border/40 pt-2 font-mono text-[9px] text-slate-400 leading-normal">
              {systemLogs.map((log, i) => (
                <div key={i}>{log}</div>
              ))}
              <div ref={terminalEndRef} />
            </div>

          </div>
        </div>

      </footer>
    </div>
  );
}
