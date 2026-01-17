
import React from 'react';
import { DiceValue } from '../types';

interface DiceProps {
  value: DiceValue;
  isKept: boolean;
  isRolling?: boolean;
  onClick?: () => void;
  disabled?: boolean;
}

const Dice: React.FC<DiceProps> = ({ value, isKept, isRolling, onClick, disabled }) => {
  const dotPositions: Record<number, string[]> = {
    1: ['center'],
    2: ['top-right', 'bottom-left'],
    3: ['top-right', 'center', 'bottom-left'],
    4: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
    5: ['top-left', 'top-right', 'center', 'bottom-left', 'bottom-right'],
    6: ['top-left', 'top-right', 'center-left', 'center-right', 'bottom-left', 'bottom-right']
  };

  const getDotStyle = (pos: string) => {
    switch (pos) {
      case 'top-left': return 'top-3 left-3';
      case 'top-right': return 'top-3 right-3';
      case 'center-left': return 'top-1/2 left-3 -translate-y-1/2';
      case 'center-right': return 'top-1/2 right-3 -translate-y-1/2';
      case 'center': return 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2';
      case 'bottom-left': return 'bottom-3 left-3';
      case 'bottom-right': return 'bottom-3 right-3';
      default: return '';
    }
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || isRolling}
      className={`
        relative w-20 h-20 rounded-[1.5rem] transition-all duration-500 ease-out
        flex items-center justify-center border-2 group
        ${isRolling ? 'animate-dice cursor-wait scale-90 opacity-80' : 'hover:scale-105'}
        ${isKept 
          ? 'bg-[#1a1a20] border-[#00ff9d] glow-green z-10' 
          : 'bg-[#141417] border-[#2d2d33] hover:border-[#4b5563]'
        }
        ${disabled && !isKept ? 'opacity-30 grayscale cursor-not-allowed scale-95' : 'cursor-pointer'}
      `}
    >
      {/* Subtle Inner Glow for Depth */}
      <div className={`absolute inset-0.5 rounded-[1.3rem] opacity-10 pointer-events-none bg-gradient-to-br ${isKept ? 'from-[#00ff9d] to-transparent' : 'from-white to-transparent'}`}></div>
      
      {!isRolling && dotPositions[value].map((pos, idx) => (
        <div
          key={idx}
          className={`absolute w-3 h-3 rounded-full transition-all duration-500 shadow-sm 
            ${isKept ? 'bg-[#00ff9d] scale-110 shadow-[#00ff9d]/20' : 'bg-slate-100'} ${getDotStyle(pos)}`}
        />
      ))}
      
      {isRolling && (
        <div className="w-4 h-4 rounded-full bg-slate-500/20 blur-[2px]"></div>
      )}
    </button>
  );
};

export default Dice;
