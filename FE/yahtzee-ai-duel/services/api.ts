
import { Scorecard, DiceValue, ScoreCategory, AIRootResponse, AIDifficulty } from '../types';

/**
 * Note: Since we don't have a live FastAPI server running in this environment,
 * these functions demonstrate how to connect to your existing .h5 model via FastAPI.
 * Replace BASE_URL with your actual local server address.
 */
const BASE_URL = 'http://localhost:8000';

export const getAIMove = async (
  dice: DiceValue[], 
  rollsRemaining: number, 
  scorecard: Scorecard,
  difficulty: AIDifficulty
): Promise<AIRootResponse> => {
  try {
    const response = await fetch(`${BASE_URL}/ai-move`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dice,
        rolls_remaining: rollsRemaining,
        scorecard,
        difficulty
      }),
    });

    if (!response.ok) throw new Error('API request failed');
    return await response.json();
  } catch (error) {
    console.warn('API Unavailable. Falling back to local simulation logic.');
    return simulateAIMove(dice, rollsRemaining, scorecard, difficulty);
  }
};

/**
 * Fallback logic if the FastAPI server isn't reachable.
 * Heuristics adjust slightly based on selected difficulty.
 */
const simulateAIMove = (
  dice: DiceValue[], 
  rollsRemaining: number, 
  scorecard: Scorecard,
  difficulty: AIDifficulty
): AIRootResponse => {
  const counts = new Array(7).fill(0);
  dice.forEach(d => counts[d]++);
  
  const mostFrequent = counts.indexOf(Math.max(...counts));
  
  // Difficulty heuristic: Novice keeps less effectively
  let diceToKeep: number[] = [];
  if (difficulty !== 'novice') {
    diceToKeep = dice.map((d, i) => d === mostFrequent ? i : -1).filter(i => i !== -1);
  } else {
    // Novice just keeps the first pair it sees or nothing
    const pair = dice.findIndex((d, i) => dice.indexOf(d) !== i);
    if (pair !== -1) diceToKeep = [dice.indexOf(dice[pair]), pair];
  }

  const categories: ScoreCategory[] = [
    'yahtzee', 'largeStraight', 'smallStraight', 'fullHouse', 
    'fourOfAKind', 'threeOfAKind', 'sixes', 'fives', 'fours', 
    'threes', 'twos', 'ones', 'chance'
  ];
  
  const availableCategory = categories.find(cat => scorecard[cat] === undefined) || 'chance';

  return {
    score_category: availableCategory,
    dice_to_keep: rollsRemaining > 0 ? diceToKeep : []
  };
};
