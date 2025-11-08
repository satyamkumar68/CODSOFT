package com.example.tictactoe_ai;

import java.util.List;

import org.springframework.stereotype.Component;

@Component
public class MinimaxAI {

    private static final int MAX_SCORE = 10;
    private static final int MIN_SCORE = -10;
    private static final int DRAW_SCORE = 0;

    public int[] findBestMove(Board board) {
        int bestVal = Integer.MIN_VALUE;
        int[] bestMove = {-1, -1};

        List<int[]> availableMoves = board.getAvailableMoves();

        for (int[] move : availableMoves) {
            // 1. Make the move on a cloned board
            Board nextBoard = deepCloneBoard(board);
            nextBoard.makeMove(move[0], move[1], 'O');

            int moveVal = minimax(nextBoard, 0, false); 

            if (moveVal > bestVal) {
                bestVal = moveVal;
                bestMove[0] = move[0];
                bestMove[1] = move[1];
            }
        }
        return bestMove;
    }

    private int minimax(Board board, int depth, boolean isMaximizingPlayer) {
        Board.GameStatus status = board.checkStatus();

        if (status != Board.GameStatus.IN_PROGRESS) {
            if (status == Board.GameStatus.O_WINS) return MAX_SCORE - depth;
            if (status == Board.GameStatus.X_WINS) return MIN_SCORE + depth;
            return DRAW_SCORE; 
        }

        List<int[]> availableMoves = board.getAvailableMoves();
        
        if (isMaximizingPlayer) { // AI's turn ('O')
            int maxEval = Integer.MIN_VALUE;
            for (int[] move : availableMoves) {
                Board nextBoard = deepCloneBoard(board);
                nextBoard.makeMove(move[0], move[1], 'O');
                int eval = minimax(nextBoard, depth + 1, false);
                maxEval = Math.max(maxEval, eval);
            }
            return maxEval;
        } else { // Human's turn ('X')
            int minEval = Integer.MAX_VALUE;
            for (int[] move : availableMoves) {
                Board nextBoard = deepCloneBoard(board);
                nextBoard.makeMove(move[0], move[1], 'X');
                int eval = minimax(nextBoard, depth + 1, true);
                minEval = Math.min(minEval, eval);
            }
            return minEval;
        }
    }

    
    private Board deepCloneBoard(Board original) {
        
        return new Board(original.cells);
    }
}
