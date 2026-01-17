
import React from 'react';
import { Scorecard, ScoreCategory } from '../types';
import { SCORING_LABELS, UPPER_SECTION, LOWER_SECTION, calculateScore } from '../constants';

interface ScoreBoardProps {
  playerScorecard: Scorecard;
  aiScorecard: Scorecard;
  currentDice: number[];
  isPlayerTurn: boolean;
  onSelectCategory: (category: ScoreCategory) => void;
  disabled: boolean;
}

const ScoreBoard: React.FC<ScoreBoardProps> = ({ 
  playerScorecard, 
  aiScorecard, 
  currentDice, 
  isPlayerTurn, 
  onSelectCategory, 
  disabled 
}) => {
  const renderRow = (category: ScoreCategory) => {
    const pScore = playerScorecard[category];
    const aiScore = aiScorecard[category];
    const potential = calculateScore(category, currentDice);
    const isAvailable = pScore === undefined && isPlayerTurn && !disabled;

    return (
      <tr key={category} className="border-b border-white/[0.01] group hover:bg-white/[0.01] transition-colors">
        <td className="py-4 px-5">
          <span className="text-[10px] font-semibold text-slate-600 uppercase tracking-[0.15em] block leading-none">
            {SCORING_LABELS[category]}
          </span>
        </td>
        <td className="py-4 px-5 text-center">
          {isAvailable ? (
            <button
              onClick={() => onSelectCategory(category)}
              className="mono text-neutral-300 font-bold hover:text-[#00d9e6] transition-all text-sm px-2 py-1 rounded bg-white/[0.02] hover:bg-[#00d9e6]/10"
            >
              {potential}
            </button>
          ) : (
            <span className={`mono text-sm ${pScore !== undefined ? 'text-neutral-200 font-bold' : 'text-neutral-900'}`}>
              {pScore ?? '-'}
            </span>
          )}
        </td>
        <td className="py-4 px-5 text-center">
          <span className={`mono text-sm ${aiScore !== undefined ? 'text-[#e6a500]/70 font-bold' : 'text-neutral-900'}`}>
            {aiScore ?? '-'}
          </span>
        </td>
      </tr>
    );
  };

  const getSectionTotal = (scorecard: Scorecard, categories: ScoreCategory[]) => {
    return categories.reduce((sum, cat) => sum + (scorecard[cat] ?? 0), 0);
  };

  const pUpper = getSectionTotal(playerScorecard, UPPER_SECTION);
  const aiUpper = getSectionTotal(aiScorecard, UPPER_SECTION);
  const pBonus = pUpper >= 63 ? 35 : 0;
  const aiBonus = aiUpper >= 63 ? 35 : 0;
  const pLower = getSectionTotal(playerScorecard, LOWER_SECTION);
  const aiLower = getSectionTotal(aiScorecard, LOWER_SECTION);

  const pTotal = pUpper + pBonus + pLower;
  const aiTotal = aiUpper + aiBonus + aiLower;

  return (
    <div className="glass-surface rounded-[2rem] overflow-hidden">
      <table className="w-full text-left border-collapse">
        <thead>
          <tr className="bg-white/[0.02] border-b border-white/[0.03]">
            <th className="py-5 px-6 text-[9px] font-black text-slate-600 tracking-[0.25em]">PARAMETER</th>
            <th className="py-5 px-6 text-center text-[9px] font-black text-neutral-400 tracking-[0.25em]">PLAYER</th>
            <th className="py-5 px-6 text-center text-[9px] font-black text-neutral-700 tracking-[0.25em]">OPPONENT</th>
          </tr>
        </thead>
        <tbody>
          {UPPER_SECTION.map(renderRow)}
          <tr className="bg-white/[0.01] border-b border-white/[0.02]">
            <td className="py-3 px-5">
              <span className="text-[9px] font-bold text-neutral-600 tracking-[0.2em]">UPPER BONUS</span>
            </td>
            <td className="py-3 px-5 text-center mono text-sm text-[#00e68e]/50">{pBonus}</td>
            <td className="py-3 px-5 text-center mono text-sm text-neutral-800">{aiBonus}</td>
          </tr>
          {LOWER_SECTION.map(renderRow)}
        </tbody>
        <tfoot>
          <tr className="bg-white/[0.01] border-t border-white/[0.02]">
            <td className="py-8 px-6">
              <span className="text-[11px] font-black text-neutral-400 uppercase tracking-[0.3em]">Aggregate</span>
            </td>
            <td className="py-8 px-6 text-center">
              <span className="text-3xl font-black text-neutral-200 mono tracking-tighter">{pTotal}</span>
            </td>
            <td className="py-8 px-6 text-center">
              <span className="text-3xl font-black text-neutral-800 mono tracking-tighter">{aiTotal}</span>
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
};

export default ScoreBoard;
