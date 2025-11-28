"""
Claude SDK Agent Builder - Complete Implementation Guide
This guide shows you how to build various types of agents using Anthropic's SDK
"""

# ============================================================================
# INSTALLATION
# ============================================================================
"""
pip install anthropic
pip install python-dotenv  # for API key management
"""

# ============================================================================
# 1. BASIC AGENT - Simple Conversational Agent
# ============================================================================

import anthropic
import os

# Initialize the client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

def basic_agent(user_message, conversation_history=None):
    """
    A basic conversational agent that maintains context
    """
    if conversation_history is None:
        conversation_history = []
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    
    # Get response from Claude
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=conversation_history
    )
    
    # Add assistant response to history
    assistant_message = response.content[0].text
    conversation_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message, conversation_history
# ============================================================================
# 2. TOOL-USING AGENT - Agent with Function Calling
# ============================================================================

def tool_using_agent():
    """
    An agent that can use tools/functions to accomplish tasks
    """
    
    # Define available tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    ]
    
    # Tool implementations
    def get_weather(location):
        # In production, call actual weather API
        return f"The weather in {location} is sunny, 72°F"
    
    def calculator(expression):
        try:
            return str(eval(expression))
        except:
            return "Invalid expression"
    
    tool_functions = {
        "get_weather": get_weather,
        "calculator": calculator
    }
    
    # Agent loop
    messages = [{
        "role": "user",
        "content": "What's the weather in Paris and what's 15 * 23?"
    }]
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    
                    # Execute the tool
                    result = tool_functions[tool_name](**tool_input)
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            
        else:
            # Final answer received
            return response.content[0].text


# ============================================================================
# 3. RAG AGENT - Retrieval Augmented Generation
# ============================================================================

def rag_agent(query, knowledge_base):
    """
    An agent that retrieves relevant context before answering
    """
    
    # Simple retrieval (in production, use embeddings + vector DB)
    def retrieve_context(query, kb):
        # Mock retrieval - return most relevant documents
        relevant_docs = [doc for doc in kb if any(
            keyword.lower() in doc.lower() 
            for keyword in query.split()
        )]
        return "\n\n".join(relevant_docs[:3])
    
    context = retrieve_context(query, knowledge_base)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Use the following context to answer the question.
            
Context:
{context}

Question: {query}

Answer based on the context provided."""
        }]
    )
    
    return response.content[0].text


# ============================================================================
# 4. MULTI-STEP REASONING AGENT (Chain of Thought)
# ============================================================================

def reasoning_agent(complex_task):
    """
    Agent that breaks down complex tasks into steps
    """
    
    # Step 1: Plan the approach
    planning_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Break down this task into clear steps:
            
Task: {complex_task}

Provide a numbered list of steps needed."""
        }]
    )
    
    plan = planning_response.content[0].text
    
    # Step 2: Execute the plan
    execution_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"Break down this task into clear steps:\n\nTask: {complex_task}\n\nProvide a numbered list of steps needed."
            },
            {
                "role": "assistant",
                "content": plan
            },
            {
                "role": "user",
                "content": "Now execute each step and provide the final solution."
            }
        ]
    )
    
    return execution_response.content[0].text


# ============================================================================
# 5. STREAMING AGENT - Real-time Response
# ============================================================================

def streaming_agent(user_message):
    """
    Agent that streams responses in real-time
    """
    
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    print()  # New line after streaming


# ============================================================================
# 6. AUTONOMOUS AGENT - Self-Directed Task Completion
# ============================================================================

class AutonomousAgent:
    """
    An agent that can autonomously decide actions and complete tasks
    """
    
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.tools = self._define_tools()
    
    def _define_tools(self):
        return [
            {
                "name": "search_web",
                "description": "Search for information on the web",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "save_to_file",
                "description": "Save content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["filename", "content"]
                }
            }
        ]
    
    def execute_task(self, task, max_iterations=10):
        """
        Execute a task autonomously with multiple iterations
        """
        
        self.conversation_history = [{
            "role": "user",
            "content": f"""Complete this task: {task}
            
You can use available tools. Think step by step and use tools as needed."""
        }]
        
        for iteration in range(max_iterations):
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                tools=self.tools,
                messages=self.conversation_history
            )
            
            if response.stop_reason == "end_turn":
                # Task completed
                final_response = next(
                    (block.text for block in response.content if hasattr(block, 'text')),
                    None
                )
                return final_response
            
            elif response.stop_reason == "tool_use":
                # Execute tools and continue
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Mock tool execution
                        result = f"Tool {block.name} executed successfully"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })
        
        return "Max iterations reached"


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Basic Agent
    print("=== BASIC AGENT ===")
    response, history = basic_agent("Hello! What can you help me with?")
    print(response)
    
    # Example 2: Tool-Using Agent
    print("\n=== TOOL-USING AGENT ===")
    result = tool_using_agent()
    print(result)
    
    # Example 3: RAG Agent
    print("\n=== RAG AGENT ===")
    kb = [
        "Our company was founded in 2020.",
        "We offer cloud services and AI solutions.",
        "Our headquarters is in San Francisco."
    ]
    answer = rag_agent("When was the company founded?", kb)
    print(answer)
    
    # Example 4: Reasoning Agent
    print("\n=== REASONING AGENT ===")
    solution = reasoning_agent("Plan a week-long trip to Japan")
    print(solution)
    
    # Example 5: Streaming Agent
    print("\n=== STREAMING AGENT ===")
    streaming_agent("Write a short poem about AI")
    
    # Example 6: Autonomous Agent
    print("\n=== AUTONOMOUS AGENT ===")
    agent = AutonomousAgent(os.environ.get("ANTHROPIC_API_KEY"))
    result = agent.execute_task("Research recent AI developments and summarize")
    print(result)
