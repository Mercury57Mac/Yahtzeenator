
import { ScoreCategory } from './types';

export const SCORING_LABELS: Record<ScoreCategory, string> = {
  ones: 'Ones',
  twos: 'Twos',
  threes: 'Threes',
  fours: 'Fours',
  fives: 'Fives',
  sixes: 'Sixes',
  threeOfAKind: '3 of a Kind',
  fourOfAKind: '4 of a Kind',
  fullHouse: 'Full House',
  smallStraight: 'Sm Straight',
  largeStraight: 'Lg Straight',
  yahtzee: 'Yahtzee',
  chance: 'Chance'
};

export const UPPER_SECTION: ScoreCategory[] = ['ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];
export const LOWER_SECTION: ScoreCategory[] = ['threeOfAKind', 'fourOfAKind', 'fullHouse', 'smallStraight', 'largeStraight', 'yahtzee', 'chance'];

export const CATEGORIES: ScoreCategory[] = [...UPPER_SECTION, ...LOWER_SECTION];

export const calculateScore = (category: ScoreCategory, dice: number[]): number => {
  const counts = new Array(7).fill(0);
  dice.forEach(d => counts[d]++);
  const sum = dice.reduce((a, b) => a + b, 0);

  switch (category) {
    case 'ones': return counts[1] * 1;
    case 'twos': return counts[2] * 2;
    case 'threes': return counts[3] * 3;
    case 'fours': return counts[4] * 4;
    case 'fives': return counts[5] * 5;
    case 'sixes': return counts[6] * 6;
    case 'threeOfAKind': return counts.some(c => c >= 3) ? sum : 0;
    case 'fourOfAKind': return counts.some(c => c >= 4) ? sum : 0;
    case 'fullHouse': {
      const hasThree = counts.some(c => c === 3);
      const hasTwo = counts.some(c => c === 2);
      const hasYahtzee = counts.some(c => c === 5);
      return (hasThree && hasTwo) || hasYahtzee ? 25 : 0;
    }
    case 'smallStraight': {
      const unique = Array.from(new Set(dice)).sort();
      let consecutive = 1;
      let maxConsecutive = 1;
      for (let i = 0; i < unique.length - 1; i++) {
        if (unique[i + 1] === unique[i] + 1) consecutive++;
        else consecutive = 1;
        maxConsecutive = Math.max(maxConsecutive, consecutive);
      }
      return maxConsecutive >= 4 ? 30 : 0;
    }
    case 'largeStraight': {
      const unique = Array.from(new Set(dice)).sort();
      if (unique.length < 5) return 0;
      let consecutive = 1;
      for (let i = 0; i < unique.length - 1; i++) {
        if (unique[i + 1] === unique[i] + 1) consecutive++;
        else consecutive = 1;
      }
      return consecutive === 5 ? 40 : 0;
    }
    case 'yahtzee': return counts.some(c => c === 5) ? 50 : 0;
    case 'chance': return sum;
    default: return 0;
  }
};
