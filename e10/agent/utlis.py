import asyncio
from agent.human_intervention import HumanInterventionError

async def show_input_dialog(prompt: str) -> str:
    """Show an input dialog in the CLI and get user input.
    
    Args:
        prompt: The prompt to show to the user
        
    Returns:
        str: The user's input
    """
    print("\n" + "="*80)
    print("🔧 Human Intervention Required")
    print("="*80)
    print(prompt)
    print("="*80)
    
    while True:
        try:
            # Use asyncio.get_event_loop().run_in_executor to handle input in a non-blocking way
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, lambda: input("\nEnter your response (or 'cancel' to abort): ").strip())
            
            if user_input.lower() == 'cancel':
                raise HumanInterventionError("Human input was cancelled", "cancelled")
            if not user_input:
                print("Response cannot be empty. Please try again.")
                continue
            return user_input
        except KeyboardInterrupt:
            raise HumanInterventionError("Human input was cancelled", "cancelled")