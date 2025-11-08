package com.example.tictactoe_ai;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Board {
    public char[][] cells;

    public enum GameStatus {
        IN_PROGRESS, X_WINS, O_WINS, DRAW
    }

    public Board() {
        cells = new char[3][3];
        for (int i = 0; i < 3; i++) {
            Arrays.fill(cells[i], ' ');
        }
    }

    public Board(char[][] cells) {
        this.cells = new char[3][3];
        for (int i = 0; i < 3; i++) {
            System.arraycopy(cells[i], 0, this.cells[i], 0, 3);
        }
    }

    public boolean isMoveValid(int row, int col) {
        if (row < 0 || row >= 3 || col < 0 || col >= 3) {
            return false;
        }
        return cells[row][col] == ' ';
    }

    public void makeMove(int row, int col, char player) {
        if (isMoveValid(row, col)) {
            cells[row][col] = player;
        }
    }

    private boolean checkLine(char c1, char c2, char c3, char player) {
        return c1 == player && c2 == player && c3 == player;
    }

    private boolean checkWin(char player) {
       
        for (int i = 0; i < 3; i++) {
            if (checkLine(cells[i][0], cells[i][1], cells[i][2], player) || // rows
                checkLine(cells[0][i], cells[1][i], cells[2][i], player)) { // columns
                return true;
            }
        }
       
        if (checkLine(cells[0][0], cells[1][1], cells[2][2], player) ||
            checkLine(cells[0][2], cells[1][1], cells[2][0], player)) {
            return true;
        }
        return false;
    }
    
    private boolean isBoardFull() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (cells[i][j] == ' ') {
                    return false;
                }
            }
        }
        return true;
    }

    public GameStatus checkStatus() {
        if (checkWin('X')) {
            return GameStatus.X_WINS;
        }
        if (checkWin('O')) {
            return GameStatus.O_WINS;
        }
        if (isBoardFull()) {
            return GameStatus.DRAW;
        }
        return GameStatus.IN_PROGRESS;
    }

    public List<int[]> getAvailableMoves() {
        List<int[]> moves = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (cells[i][j] == ' ') {
                    moves.add(new int[]{i, j});
                }
            }
        }
        return moves;
    }

    public char[][] getCells() {
        return cells;
    }
}
