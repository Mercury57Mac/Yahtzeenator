
export type ScoreCategory = 
  | 'ones' | 'twos' | 'threes' | 'fours' | 'fives' | 'sixes'
  | 'threeOfAKind' | 'fourOfAKind' | 'fullHouse' 
  | 'smallStraight' | 'largeStraight' | 'yahtzee' | 'chance';

export type AIDifficulty = 'novice' | 'strategic' | 'neural';

export interface Scorecard {
  ones?: number;
  twos?: number;
  threes?: number;
  fours?: number;
  fives?: number;
  sixes?: number;
  threeOfAKind?: number;
  fourOfAKind?: number;
  fullHouse?: number;
  smallStraight?: number;
  largeStraight?: number;
  yahtzee?: number;
  chance?: number;
}

export type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

export interface GameState {
  dice: DiceValue[];
  keptDice: boolean[];
  rollsRemaining: number;
  playerScorecard: Scorecard;
  aiScorecard: Scorecard;
  isPlayerTurn: boolean;
  gameStarted: boolean;
  gameOver: boolean;
  aiThinking: boolean;
  statusMessage: string;
  difficulty: AIDifficulty;
}

export interface AIRootResponse {
  score_category: ScoreCategory;
  dice_to_keep: number[];
}
