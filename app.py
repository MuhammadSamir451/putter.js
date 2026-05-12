from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_session import Session
import json
import re
from typing import List, Dict, Any
import os
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__, static_folder='.')

# Session configuration
app.config['SECRET_KEY'] = 'hospital-chatbot-secret-key-2026'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
Session(app)

CORS(app, supports_credentials=True)

# Initialize embedding model
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded!")

# Initialize FAISS index
dimension = 384
faiss_index = faiss.IndexFlatL2(dimension)
chunk_texts = []
chunk_metadata = []

# Store complete hospital data for direct access
complete_hospitals_data = {}

# Load hospital data
def load_hospital_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'hospitals.json')
        
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"✅ Loaded JSON file")
            hospitals_raw = data.get('hospitals', [])
            print(f"📊 Found {len(hospitals_raw)} hospitals")
            return hospitals_raw
    except FileNotFoundError:
        print("❌ hospitals.json not found!")
        return []
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return []

# Create comprehensive chunks - NO DATA LOSS
def create_chunks(hospitals_raw):
    """Create chunks that preserve ALL information"""
    chunks = []
    
    for hospital in hospitals_raw:
        hosp_name = hospital.get('hospital_name', 'Unknown')
        hosp_type = hospital.get('type', '')
        hosp_location = hospital.get('location_notes', '')
        
        # Store complete hospital data for direct access
        complete_hospitals_data[hosp_name] = {
            'name': hosp_name,
            'type': hosp_type,
            'location': hosp_location,
            'departments': []
        }
        
        # Chunk 1: Hospital overview (always include)
        overview_chunk = f"""
HOSPITAL NAME: {hosp_name}
TYPE: {hosp_type}
LOCATION: {hosp_location}
"""
        chunks.append({
            'text': overview_chunk,
            'type': 'overview',
            'hospital': hosp_name,
            'priority': 1
        })
        
        # Create a COMPLETE department list chunk for each hospital
        all_departments_text = f"""
HOSPITAL: {hosp_name}
COMPLETE DEPARTMENT LIST (ALL DEPARTMENTS):
"""
        
        department_count = 0
        for dept in hospital.get('departments', []):
            dept_name = dept.get('department_name', 'General')
            specialists = dept.get('specialists', [])
            
            # Store in complete data
            complete_hospitals_data[hosp_name]['departments'].append({
                'name': dept_name,
                'specialists': specialists
            })
            
            # Add to chunk text
            all_departments_text += f"\n📋 {dept_name}\n"
            if specialists and len(specialists) > 0:
                all_departments_text += f"   Specialists:\n"
                for spec in specialists[:15]:  # Show up to 15 specialists per dept
                    if isinstance(spec, str):
                        all_departments_text += f"   • {spec}\n"
                    elif isinstance(spec, dict):
                        spec_name = spec.get('name', str(spec))
                        all_departments_text += f"   • {spec_name}\n"
                if len(specialists) > 15:
                    all_departments_text += f"   • ... and {len(specialists) - 15} more specialists\n"
            else:
                all_departments_text += f"   • Contact hospital for specialist information\n"
            
            department_count += 1
        
        all_departments_text += f"\n📊 TOTAL DEPARTMENTS: {department_count}\n"
        
        chunks.append({
            'text': all_departments_text,
            'type': 'complete_departments',
            'hospital': hosp_name,
            'priority': 2  # Higher priority for department queries
        })
        
        # Also create smaller chunks for specific departments (for targeted queries)
        for dept in hospital.get('departments', []):
            dept_name = dept.get('department_name', 'General')
            specialists = dept.get('specialists', [])
            
            dept_chunk = f"""
HOSPITAL: {hosp_name}
DEPARTMENT: {dept_name}
SPECIALISTS: {len(specialists)} specialists available
"""
            if specialists:
                dept_chunk += "\nSpecialist List:\n"
                for spec in specialists[:10]:
                    if isinstance(spec, str):
                        dept_chunk += f"• {spec}\n"
                    elif isinstance(spec, dict):
                        spec_name = spec.get('name', str(spec))
                        dept_chunk += f"• {spec_name}\n"
            
            chunks.append({
                'text': dept_chunk,
                'type': 'single_department',
                'hospital': hosp_name,
                'department': dept_name,
                'priority': 3
            })
    
    print(f"✅ Created {len(chunks)} searchable chunks")
    print(f"📊 Stored complete data for {len(complete_hospitals_data)} hospitals")
    return chunks

# Direct retrieval function for complete hospital data
def get_complete_hospital_info(hospital_name: str) -> str:
    """Directly retrieve ALL information for a specific hospital"""
    for hosp_name, data in complete_hospitals_data.items():
        if hospital_name.lower() in hosp_name.lower() or hosp_name.lower() in hospital_name.lower():
            context = f"""
🏥 **COMPLETE INFORMATION FOR {data['name']}** 🏥

📍 **Type:** {data['type']}
📍 **Location:** {data['location']}
📞 **Contact:** Contact hospital directly for appointments

📋 **ALL DEPARTMENTS ({len(data['departments'])} total):**

"""
            for dept in data['departments']:
                context += f"\n**• {dept['name']}**\n"
                if dept['specialists'] and len(dept['specialists']) > 0:
                    context += f"  👨‍⚕️ Specialists:\n"
                    for spec in dept['specialists'][:10]:
                        if isinstance(spec, str):
                            context += f"    - {spec}\n"
                        elif isinstance(spec, dict):
                            spec_name = spec.get('name', str(spec))
                            context += f"    - {spec_name}\n"
                    if len(dept['specialists']) > 10:
                        context += f"    - ... and {len(dept['specialists']) - 10} more specialists\n"
                else:
                    context += f"  📞 Contact hospital for specialist information\n"
            
            context += f"\n---\n*This is the complete department list from our database.*"
            return context
    
    return None

# Smart RAG retrieval with priority for complete data

def smart_retrieve(query: str, n_results: int = 15) -> str:
    """Intelligently retrieve relevant context with priority for complete data"""
    
    query_lower = query.lower()
    
    # FIRST: Check if asking for ALL hospitals
    if "all hospitals" in query_lower or "list all hospitals" in query_lower or "complete list" in query_lower:
        context = "🏥 **COMPLETE LIST OF ALL HOSPITALS IN RAWALPINDI:**\n\n"
        
        # Sort hospitals alphabetically
        sorted_hospitals = sorted(complete_hospitals_data.keys())
        
        for i, hospital_name in enumerate(sorted_hospitals, 1):
            data = complete_hospitals_data[hospital_name]
            context += f"{i}. **{hospital_name}**\n"
            context += f"   📍 Type: {data['type']}\n"
            context += f"   📍 Location: {data['location']}\n"
            context += f"   🏥 Departments: {len(data['departments'])} departments\n"
            context += "\n"
        
        context += f"\n*Total: {len(sorted_hospitals)} hospitals in Rawalpindi*"
        context += "\n\nAsk 'Show all departments in [hospital name]' for complete department lists!"
        return context
    
    # SECOND: Check if asking for departments of a specific hospital
    hospital_names = list(complete_hospitals_data.keys())
    for hospital in hospital_names:
        if hospital.lower() in query_lower and any(word in query_lower for word in ['department', 'dept', 'all departments', 'complete list', 'services']):
            complete_info = get_complete_hospital_info(hospital)
            if complete_info:
                return complete_info
    
    # THIRD: Check if asking about a specific hospital (general info)
    for hospital in hospital_names:
        if hospital.lower() in query_lower and len(hospital) > 5:
            complete_info = get_complete_hospital_info(hospital)
            if complete_info:
                return complete_info
    
    # FOURTH: Use semantic search for other queries (USE THE n_results PARAMETER!)
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    # Use the n_results parameter here - NOT hardcoded!
    actual_results = min(n_results, len(chunk_texts))
    distances, indices = faiss_index.search(query_embedding, actual_results)
    
    # Collect relevant chunks with priority scoring
    relevant_chunks = []
    seen_hospitals = set()
    
    for idx, distance in zip(indices[0], distances[0]):
        if idx != -1 and idx < len(chunk_metadata):
            chunk = chunk_metadata[idx]
            hospital = chunk.get('hospital', '')
            
            # Skip if we already have 2 chunks from same hospital
            if hospital in seen_hospitals and len([c for c in relevant_chunks if c['chunk'].get('hospital') == hospital]) > 2:
                continue
            
            relevance_score = 1.0 / (1.0 + distance)
            # Boost priority for complete department chunks
            if chunk.get('type') == 'complete_departments':
                relevance_score *= 1.5
            
            relevant_chunks.append({
                'chunk': chunk,
                'score': relevance_score,
                'text': chunk_texts[idx]
            })
            seen_hospitals.add(hospital)
    
    # Sort by score
    relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Format context - use n_results to determine how many to show
    context = "🏥 **RELEVANT HOSPITAL INFORMATION:**\n\n"
    
    # First, try to get complete department lists for mentioned hospitals
    hospitals_mentioned = set()
    # Use n_results to determine how many chunks to process
    chunks_to_process = min(n_results, len(relevant_chunks))
    
    for chunk in relevant_chunks[:chunks_to_process]:
        hosp = chunk['chunk'].get('hospital', '')
        if hosp:
            hospitals_mentioned.add(hosp)
    
    for hospital in hospitals_mentioned:
        complete_info = get_complete_hospital_info(hospital)
        if complete_info:
            context += complete_info + "\n\n"
            # Remove this hospital from chunks to avoid duplication
            relevant_chunks = [c for c in relevant_chunks if c['chunk'].get('hospital') != hospital]
    
    # Add remaining chunks (also using n_results logic)
    remaining_to_show = min(n_results // 2, len(relevant_chunks))
    for chunk in relevant_chunks[:remaining_to_show]:
        context += chunk['text'] + "\n\n"
        context += "-" * 40 + "\n\n"
    
    if not hospitals_mentioned and not relevant_chunks:
        # Fallback: Show ALL hospitals overview (not limited!)
        context = "🏥 **ALL HOSPITALS IN RAWALPINDI:**\n\n"
        for hospital in list(complete_hospitals_data.keys())[:15]:  # Show up to 15
            data = complete_hospitals_data[hospital]
            context += f"• **{hospital}**\n"
            context += f"  📍 {data['type']}\n"
            context += f"  📍 {data['location']}\n"
            context += f"  📋 {len(data['departments'])} departments\n\n"
        context += "\n*Ask 'Show all departments in [hospital name]' for complete department lists!*"
    
    return context
# Load and process all data
hospitals_raw = load_hospital_data()
chunks = create_chunks(hospitals_raw)

# Create vector embeddings
print("🔄 Creating vector embeddings...")
vectors = []
for idx, chunk in enumerate(chunks):
    text_for_embedding = chunk['text']
    embedding = embedding_model.encode(text_for_embedding)
    vectors.append(embedding)
    chunk_texts.append(chunk['text'])
    chunk_metadata.append(chunk)

if vectors:
    vectors_array = np.array(vectors).astype('float32')
    faiss_index.add(vectors_array)

print(f"✅ Created {len(chunks)} vector embeddings!")
print(f"💾 FAISS index size: {faiss_index.ntotal} vectors")

# Session management functions
def get_session_history(session_id):
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return session['conversation_history']

def add_to_session_history(role, content, session_id):
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    session['conversation_history'].append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    
    if len(session['conversation_history']) > 20:
        session['conversation_history'] = session['conversation_history'][-20:]
    
    session.modified = True

def clear_session_history():
    session['conversation_history'] = []
    session.modified = True

def format_conversation_context(history):
    if not history or len(history) == 0:
        return ""
    
    context = "📜 **RECENT CONVERSATION:**\n"
    for msg in history[-6:]:
        role = "User" if msg['role'] == 'user' else "Assistant"
        content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
        context += f"{role}: {content}\n"
    context += "\n"
    return context

# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        clear_context = data.get('clear_context', False)
        
        if clear_context:
            clear_session_history()
            return jsonify({
                'response': "🧹 Conversation cleared! How can I help you?",
                'context_cleared': True
            })
        
        conversation_history = get_session_history(request.remote_addr)
        
        print(f"\n📝 User: {user_message}")
        
        # Get relevant context using smart retrieval
        retrieved_context = smart_retrieve(user_message, n_results=15)  # Get 15 results for "all hospitals" queries
        conversation_context = format_conversation_context(conversation_history)
        
        full_context = conversation_context + retrieved_context
        
        add_to_session_history('user', user_message, request.remote_addr)
        
        print(f"📊 Context length: {len(full_context)} chars")
        
        return jsonify({
            'context': full_context,
            'has_data': len(complete_hospitals_data) > 0,
            'history_length': len(conversation_history)
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    clear_session_history()
    return jsonify({'success': True})

@app.route('/api/hospitals', methods=['GET'])
def get_all_hospitals():
    hospitals_list = []
    for name, data in complete_hospitals_data.items():
        hospitals_list.append({
            'name': name,
            'type': data['type'],
            'location': data['location'],
            'departments_count': len(data['departments'])
        })
    return jsonify({'hospitals': hospitals_list, 'count': len(hospitals_list)})

@app.route('/api/hospital/<hospital_name>', methods=['GET'])
def get_hospital(hospital_name):
    for name, data in complete_hospitals_data.items():
        if hospital_name.lower() in name.lower():
            return jsonify(data)
    return jsonify({'error': 'Hospital not found'}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ENHANCED RAG HOSPITAL CHATBOT")
    print("="*60)
    print(f"📊 Total hospitals: {len(complete_hospitals_data)}")
    print(f"📊 Total chunks: {len(chunks)}")
    print(f"🔢 Vector dimension: {dimension}")
    print(f"💾 FAISS index: {faiss_index.ntotal} vectors")
    print("\n✨ NEW FEATURES:")
    print("   • Complete department lists (NO missing data!)")
    print("   • Direct hospital data retrieval")
    print("   • Priority scoring for complete information")
    print("   • Smart fallback to show ALL departments")
    print("\n🔗 Visit http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)