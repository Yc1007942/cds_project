import sys
from pathlib import Path

# Add project root to sys.path
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ml_model import get_model

def test_prediction():
    print("🔍 Testing Model Integration...")
    model = get_model()
    
    post_text = "What is the meaning of life in the age of artificial intelligence? is it a question of ethics or technology?"
    forum = "philosophy"
    
    print(f"\n1. Testing INITIAL prediction (no comments) for forum: {forum}")
    initial_score = model.predict_score(post_text, forum=forum)
    print(f"✅ Initial Score: {initial_score:.2f}")
    
    print("\n2. Testing UPDATED prediction (with simulated agent responses)")
    mock_responses = [
        {"agentName": "Socrates", "response": "I think we must first define what life even means in this context. It is deeply ethical."},
        {"agentName": "TechEnthusiast", "response": "It's all about the data and the processing power. Ethics is just a subroutine."},
        {"agentName": "Existentialist", "response": "The meaning is what we create, regardless of the substrate of intelligence."}
    ]
    
    updated_score = model.predict_score(post_text, forum=forum, comments=mock_responses)
    print(f"✅ Updated Score: {updated_score:.2f}")
    
    if updated_score != initial_score:
        print("\n🎉 SUCCESS: The score changed after adding comments!")
    else:
        print("\n⚠️ NOTE: The score did not change. This could happen if the model is a placeholder or comments have neutral sentiment.")

if __name__ == "__main__":
    try:
        test_prediction()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
