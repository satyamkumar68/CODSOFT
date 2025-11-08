package com.example.tictactoe_ai;

import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/game")
public class GameController {

    private final MinimaxAI ai;
    private Board board; 

    @Autowired
    public GameController(MinimaxAI ai) {
        this.ai = ai;
        this.board = new Board();
    }

    private Map<String, Object> getBoardResponse() {
        Map<String, Object> response = new HashMap<>();
        char[][] cells = board.getCells();
        String[][] boardForJson = new String[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                boardForJson[i][j] = String.valueOf(cells[i][j]);
            }
        }
        response.put("board", boardForJson);
        response.put("status", board.checkStatus().name());
        return response;
    }

    @PostMapping("/new")
    public ResponseEntity<Map<String, Object>> startNewGame() {
        board = new Board(); 
        return ResponseEntity.ok(getBoardResponse());
    }

    @PostMapping("/move/{row}/{col}")
    public synchronized ResponseEntity<Map<String, Object>> makeMove(@PathVariable int row, @PathVariable int col) {
        if (board == null || board.checkStatus() != Board.GameStatus.IN_PROGRESS) {
            return ResponseEntity.badRequest().body(Map.of("message", "Game is not active. Start a new game first."));
        }

        if (!board.isMoveValid(row, col)) {
            return ResponseEntity.badRequest().body(Map.of("message", "Invalid move. Cell is already occupied or coordinates are out of bounds."));
        }

        board.makeMove(row, col, 'X');
        Board.GameStatus status = board.checkStatus();

        if (status == Board.GameStatus.IN_PROGRESS) {
            // 2. AI Move ('O')
            int[] aiMove = ai.findBestMove(board);
            if (aiMove[0] != -1) {
                board.makeMove(aiMove[0], aiMove[1], 'O');
            }
        }

        return ResponseEntity.ok(getBoardResponse());
    }

    @GetMapping("/status")
    public Map<String, Object> getStatus() {
        if (board == null) {
            board = new Board();
        }
        return getBoardResponse();
    }
}
