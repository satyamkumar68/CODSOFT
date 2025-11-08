package com.chatbot.rule_based_chatbot;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

import jakarta.servlet.http.HttpSession;

@Controller
public class ChatController {

    private static final String MESSAGE_HISTORY_KEY = "chatHistory";

    
    public static class Message {
        private String sender;
        private String text;

        public Message(String sender, String text) {
            this.sender = sender;
            this.text = text;
        }
        public String getSender() { return sender; }
        public String getText() { return text; }
    }

    
    private String generateResponse(String input) {
        
        String lowerInput = input.toLowerCase().trim();


        Pattern greetingPattern = Pattern.compile("^(hi|hello|hey|good (morning|day|evening)|how are you).*", Pattern.CASE_INSENSITIVE);
        if (greetingPattern.matcher(lowerInput).matches()) {
            return "Hello! I see you're starting a conversation. I'm a rule-based Java chatbot. What's on your mind?";
        }

        Pattern definitionPattern = Pattern.compile(".*(what is|tell me about|explain|define) (java|spring boot|regex|controller).*", Pattern.CASE_INSENSITIVE);
        Matcher definitionMatcher = definitionPattern.matcher(lowerInput);
        if (definitionMatcher.matches()) {
            
            
            if (lowerInput.contains("java")) {
                return "Java is an object-oriented, class-based language designed to have as few implementation dependencies as possible. You know, 'Write Once, Run Anywhere'.";
            } else if (lowerInput.contains("spring boot")) {
                return "Spring Boot is a framework that speeds up the development of production-ready Spring applications by handling configuration for you.";
            } else if (lowerInput.contains("regex")) {
                return "Regular Expressions (Regex) are sequences of characters that define a complex search pattern for string matching, perfect for pattern matching in text!";
            } else if (lowerInput.contains("controller")) {
                return "In Spring MVC, a **Controller** handles incoming web requests, processes them, and returns a response (like our chat page!).";
            }
        }
        
        
        if (lowerInput.contains("thank") || lowerInput.contains("thanks")) {
            return "You're most welcome! I'm happy to assist with the predefined rules.";
        } 

    
        return "I'm sorry, I couldn't find a rule matching your specific query. Try asking 'What is Java?' or simply 'Hello'.";
    }

    
    @GetMapping({"/", "/chat"})
    public String handleChat(@RequestParam(name="query", required=false) String userQuery, Model model, HttpSession session) {
        
        
        @SuppressWarnings("unchecked")
        List<Message> history = (List<Message>) session.getAttribute(MESSAGE_HISTORY_KEY);
        if (history == null) {
            history = new ArrayList<>();
            
            history.add(new Message("Bot", "Welcome! I'm your Java rule-based chatbot. Let's start the conversation."));
        }

        
        if (userQuery != null && !userQuery.trim().isEmpty()) {
           
            history.add(new Message("User", userQuery));

           
            String botResponse = generateResponse(userQuery);
            history.add(new Message("Bot", botResponse));
        }

        
        session.setAttribute(MESSAGE_HISTORY_KEY, history);

        
        model.addAttribute("chatHistory", history);
        
        return "chat"; // Renders the chat.html template
    }
}