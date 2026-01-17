
import React, { useState, useEffect, useCallback, useRef } from 'react';
import Dice from './components/Dice';
import ScoreBoard from './components/ScoreBoard';
import { GameState, DiceValue, ScoreCategory, AIDifficulty } from './types';
import { calculateScore, CATEGORIES } from './constants';
import { getAIMove } from './services/api';

const INITIAL_DICE: DiceValue[] = [1, 2, 3, 4, 5];

const App: React.FC = () => {
  const [state, setState] = useState<GameState>({
    dice: INITIAL_DICE,
    keptDice: [false, false, false, false, false],
    rollsRemaining: 3,
    playerScorecard: {},
    aiScorecard: {},
    isPlayerTurn: true,
    gameStarted: false,
    gameOver: false,
    aiThinking: false,
    statusMessage: "System Ready",
    difficulty: 'neural'
  });

  const [isRolling, setIsRolling] = useState(false);
  const [log, setLog] = useState<string>("Awaiting player initiation.");
  const rollingTimeoutRef = useRef<number | null>(null);

  const getTotals = (scorecard: any) => {
    const upper = CATEGORIES.slice(0, 6).reduce((sum, cat) => sum + (scorecard[cat] ?? 0), 0);
    const bonus = upper >= 63 ? 35 : 0;
    const lower = CATEGORIES.slice(6).reduce((sum, cat) => sum + (scorecard[cat] ?? 0), 0);
    return upper + bonus + lower;
  };

  const rollDice = () => {
    if (state.rollsRemaining === 0 || state.aiThinking || isRolling) return;

    setIsRolling(true);
    setLog("Rolling sequence engaged...");

    rollingTimeoutRef.current = window.setTimeout(() => {
      const newDice = [...state.dice];
      for (let i = 0; i < 5; i++) {
        if (!state.keptDice[i]) {
          newDice[i] = (Math.floor(Math.random() * 6) + 1) as DiceValue;
        }
      }

      setState(prev => ({
        ...prev,
        dice: newDice,
        rollsRemaining: prev.rollsRemaining - 1,
        gameStarted: true,
        statusMessage: `Sequence ${3 - (prev.rollsRemaining - 1)} / 3`
      }));

      setIsRolling(false);

      if (3 - (state.rollsRemaining - 1) === 1) {
        setLog("New cycle started. Analysis pending.");
      }
    }, 700);
  };

  const toggleKeep = (index: number) => {
    if (state.rollsRemaining === 3 || !state.isPlayerTurn || state.aiThinking || isRolling) return;
    const newKept = [...state.keptDice];
    newKept[index] = !newKept[index];
    setState(prev => ({ ...prev, keptDice: newKept }));
  };

  const selectCategory = (category: ScoreCategory) => {
    if (!state.isPlayerTurn || state.playerScorecard[category] !== undefined || state.rollsRemaining === 3 || isRolling) return;

    const score = calculateScore(category, state.dice);
    const newScorecard = { ...state.playerScorecard, [category]: score };

    setState(prev => ({
      ...prev,
      playerScorecard: newScorecard,
      isPlayerTurn: false,
      rollsRemaining: 3,
      keptDice: [false, false, false, false, false],
      statusMessage: "Opponent calculating..."
    }));

    setLog(`Registered ${score} pts for ${category.toUpperCase()}. Transitioning turn.`);
    checkGameOver(newScorecard, state.aiScorecard);
  };

  const checkGameOver = (pScore: any, aiScore: any) => {
    const pDone = Object.keys(pScore).length === 13;
    const aiDone = Object.keys(aiScore).length === 13;
    if (pDone && aiDone) {
      setState(prev => ({ ...prev, gameOver: true, statusMessage: "Final State" }));
      setLog("Game cycle terminated. Analysis complete.");
    }
  };

  const performAIMove = useCallback(async () => {
    if (state.isPlayerTurn || state.gameOver || state.aiThinking) return;

    setState(prev => ({ ...prev, aiThinking: true }));
    setLog("Opponent initiating turn sequence...");

    // Simulate AI picking up dice and rolling
    await new Promise(r => setTimeout(r, 500));

    // Generate fresh dice for AI's first roll
    let currentDice = Array.from({ length: 5 }, () => (Math.floor(Math.random() * 6) + 1) as DiceValue);

    setState(prev => ({
      ...prev,
      dice: currentDice,
      keptDice: [false, false, false, false, false],
      statusMessage: "Opponent Roll 1 / 3"
    }));

    await new Promise(r => setTimeout(r, 800));

    let currentKept = [false, false, false, false, false];
    let rolls = 3;

    while (rolls > 0) {
      const decision = await getAIMove(currentDice, rolls - 1, state.aiScorecard, state.difficulty);

      if (rolls > 1) {
        currentKept = [false, false, false, false, false];
        decision.dice_to_keep.forEach(idx => { if (idx < 5) currentKept[idx] = true; });

        setState(prev => ({ ...prev, keptDice: currentKept }));
        await new Promise(r => setTimeout(r, 600));

        setIsRolling(true);
        await new Promise(r => setTimeout(r, 800));
        setIsRolling(false);

        for (let i = 0; i < 5; i++) {
          if (!currentKept[i]) {
            currentDice[i] = (Math.floor(Math.random() * 6) + 1) as DiceValue;
          }
        }

        setState(prev => ({
          ...prev,
          dice: currentDice,
          rollsRemaining: rolls - 1,
          statusMessage: `Opponent Roll ${3 - (rolls - 1)} / 3`
        }));
      } else {
        const score = calculateScore(decision.score_category, currentDice);
        const newScorecard = { ...state.aiScorecard, [decision.score_category]: score };

        await new Promise(r => setTimeout(r, 800));

        setState(prev => ({
          ...prev,
          aiScorecard: newScorecard,
          isPlayerTurn: true,
          rollsRemaining: 3,
          keptDice: [false, false, false, false, false],
          aiThinking: false,
          statusMessage: "Your Sequence"
        }));

        setLog(`Opponent allocated ${score} pts to ${decision.score_category.toUpperCase()}.`);
        checkGameOver(state.playerScorecard, newScorecard);
        break;
      }
      rolls--;
    }
  }, [state.isPlayerTurn, state.gameOver, state.dice, state.aiScorecard, state.playerScorecard, state.difficulty]);

  useEffect(() => {
    if (!state.isPlayerTurn && !state.gameOver) {
      performAIMove();
    }
  }, [state.isPlayerTurn, state.gameOver, performAIMove]);

  const pTotal = getTotals(state.playerScorecard);
  const aiTotal = getTotals(state.aiScorecard);

  const changeDifficulty = (level: AIDifficulty) => {
    if (state.gameStarted && !state.gameOver) {
      if (!confirm("Changing difficulty will reset the current game. Proceed?")) return;
      window.location.reload();
    }
    setState(prev => ({ ...prev, difficulty: level }));
    setLog(`Opponent intelligence reconfigured to ${level.toUpperCase()}.`);
  };

  return (
    <div className="min-h-screen px-6 py-12 md:px-16 md:py-16 flex flex-col items-center max-w-7xl mx-auto selection:bg-white/10">

      {/* Bot Level Selector Row */}
      <div className="w-full flex justify-center mb-12">
        <div className="glass-surface p-1 rounded-full flex gap-1 border-white/5 shadow-none">
          {(['novice', 'strategic', 'neural'] as AIDifficulty[]).map((level) => (
            <button
              key={level}
              onClick={() => changeDifficulty(level)}
              className={`px-6 py-2 rounded-full text-[9px] font-black uppercase tracking-[0.2em] transition-all duration-300 ${state.difficulty === level
                ? 'bg-neutral-700 text-white'
                : 'text-neutral-600 hover:text-neutral-400'
                }`}
            >
              {level}
            </button>
          ))}
        </div>
      </div>

      {/* Sleek Header Section */}
      <header className="w-full flex flex-col md:flex-row justify-between items-center mb-24 gap-12">
        <div className="flex items-center gap-6">
          <div className="w-1.5 h-12 bg-neutral-700 rounded-full"></div>
          <div>
            <h1 className="text-3xl font-black tracking-[-0.04em] text-white uppercase leading-none">STRATEGY ENGINE</h1>
            <p className="text-neutral-600 text-[9px] font-bold tracking-[0.4em] uppercase mt-2">
              {state.difficulty === 'neural' ? 'Neural Network Inferencing v2.0' : state.difficulty === 'strategic' ? 'Heuristic Analysis v1.4' : 'Basic Logic Routine'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-20">
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-black text-neutral-500 uppercase tracking-[0.3em] mb-4">Player Focus</span>
            <div className="mono text-5xl font-black text-white tracking-tighter">{pTotal}</div>
          </div>
          <div className="w-px h-16 bg-white/5"></div>
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-black text-neutral-500 uppercase tracking-[0.3em] mb-4 opacity-40">Opponent Node</span>
            <div className="mono text-5xl font-black text-white/20 tracking-tighter">{aiTotal}</div>
          </div>
        </div>
      </header>

      <main className="w-full grid grid-cols-1 lg:grid-cols-12 gap-16 items-start">

        {/* Gaming Arena */}
        <div className="lg:col-span-7 flex flex-col items-center space-y-12">

          {/* Console / Log Unit */}
          <div className="w-full space-y-4">
            <div className="glass-surface p-6 rounded-3xl flex items-center justify-between border-l-2 border-l-neutral-700">
              <div className="flex items-center gap-4">
                <div className={`w-2.5 h-2.5 rounded-full transition-colors duration-500 ${state.aiThinking ? 'bg-[#ffb700] animate-pulse' : 'bg-neutral-600'}`}></div>
                <span className="text-[11px] font-bold text-slate-400 uppercase tracking-widest">{state.statusMessage}</span>
              </div>
              <div className="mono text-[10px] text-neutral-600 font-medium tracking-widest uppercase">NODE: {state.difficulty}_INF</div>
            </div>

            <div className="bg-[#0c0c0f] p-6 rounded-3xl border border-white/[0.02] min-h-[100px] flex items-start shadow-inner">
              <p className="text-slate-500 font-medium text-sm leading-relaxed tracking-wide italic">
                {log}
              </p>
            </div>
          </div>

          {/* Table Surface */}
          <div className="glass-surface p-16 rounded-[4rem] w-full flex flex-col items-center gap-16 relative overflow-hidden group">
            {/* Ambient Background Gradient for the dice box */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,_rgba(255,255,255,0.01)_0%,_transparent_100%)]"></div>

            <div className="flex flex-wrap justify-center gap-8 relative z-10">
              {state.dice.map((val, idx) => (
                <div key={idx} className="flex flex-col items-center gap-4">
                  <Dice
                    value={val}
                    isKept={state.keptDice[idx]}
                    isRolling={isRolling && !state.keptDice[idx]}
                    onClick={() => toggleKeep(idx)}
                    disabled={!state.isPlayerTurn || state.rollsRemaining === 3 || state.rollsRemaining === 0}
                  />
                  <div className={`h-1 rounded-full transition-all duration-700 ${state.keptDice[idx] ? 'w-8 bg-[#00e68e] shadow-[0_0_10px_rgba(0,230,142,0.1)]' : 'w-0 bg-transparent'}`}></div>
                </div>
              ))}
            </div>

            <div className="flex flex-col items-center gap-8 w-full relative z-10">
              <button
                onClick={rollDice}
                disabled={!state.isPlayerTurn || state.rollsRemaining === 0 || state.aiThinking || state.gameOver || isRolling}
                className={`
                  w-full max-w-md py-7 rounded-3xl text-[11px] font-black uppercase tracking-[0.5em] transition-all duration-500 border
                  ${state.rollsRemaining > 0 && state.isPlayerTurn && !state.aiThinking && !isRolling
                    ? 'bg-[#1a1a1e] border-white/5 text-neutral-200 hover:bg-[#25252b] hover:border-white/10 hover:scale-[1.01] shadow-2xl active:scale-95'
                    : 'bg-[#0a0a0c] border-white/[0.01] text-neutral-800 cursor-not-allowed'}
                `}
              >
                {state.rollsRemaining === 3 ? 'Initiate Turn Cycle' : `Recalculate (${state.rollsRemaining})`}
              </button>

              {/* Muted Sequence Dots */}
              <div className="flex gap-4">
                {[...Array(3)].map((_, i) => (
                  <div
                    key={i}
                    className={`h-1.5 w-12 rounded-full transition-all duration-700 ${i < state.rollsRemaining ? 'bg-neutral-600' : 'bg-white/[0.03]'}`}
                  />
                ))}
              </div>
            </div>
          </div>

          {state.gameOver && (
            <div className="mt-16 text-center animate-in fade-in slide-in-from-bottom-4 duration-1000">
              <h2 className="text-6xl font-black mb-8 tracking-tighter uppercase italic text-neutral-200">
                {pTotal > aiTotal ? 'Dominance Achieved' : pTotal < aiTotal ? 'System Overpowered' : 'Equilibrium'}
              </h2>
              <button
                onClick={() => window.location.reload()}
                className="bg-neutral-200 text-black px-16 py-5 rounded-full font-black uppercase tracking-[0.3em] text-[10px] hover:bg-white transition-all hover:scale-105"
              >
                Re-initialize Environment
              </button>
            </div>
          )}
        </div>

        {/* Matrix Sidebar */}
        <div className="lg:col-span-5 w-full">
          <ScoreBoard
            playerScorecard={state.playerScorecard}
            aiScorecard={state.aiScorecard}
            currentDice={state.dice}
            isPlayerTurn={state.isPlayerTurn}
            onSelectCategory={selectCategory}
            disabled={state.rollsRemaining === 3 || state.aiThinking || isRolling}
          />
        </div>
      </main>

      {/* Architecture Footer */}
      <footer className="mt-40 w-full grid grid-cols-1 md:grid-cols-3 gap-16 border-t border-white/[0.05] pt-16 mb-24 opacity-60">
        <div className="space-y-4">
          <h4 className="font-black text-slate-400 uppercase text-[9px] tracking-[0.4em]">Strategic Logic</h4>
          <p className="text-slate-600 text-xs leading-relaxed font-medium">High-fidelity probabilistic computation. 13-stage categorization sequence. Adaptive weight-based optimization for turn-cycle allocation.</p>
        </div>
        <div className="space-y-4">
          <h4 className="font-black text-slate-400 uppercase text-[9px] tracking-[0.4em]">Decision Matrix</h4>
          <p className="text-slate-600 text-xs leading-relaxed font-medium">Deep inference model processed via asynchronous FastAPI endpoints. Neural weights optimized for maximal expected value outcomes.</p>
        </div>
        <div className="space-y-4">
          <h4 className="font-black text-slate-400 uppercase text-[9px] tracking-[0.4em]">UI Protocol</h4>
          <p className="text-slate-600 text-xs leading-relaxed font-medium">Minimalist charcoal-plane architecture. Muted system feedback loops for critical states. Built on React component synchronization.</p>
        </div>
      </footer>
    </div>
  );
};

export default App;
