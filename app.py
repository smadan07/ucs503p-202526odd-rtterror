import os
import jwt
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room # New imports
from functools import wraps
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from sqlalchemy.types import PickleType
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load environment variables
load_dotenv()

# --- INITIALIZATION ---
app = Flask(__name__)
# We must allow async_mode for gevent
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent') # New init
CORS(app) # CORS is already handled by SocketIO, but this is fine

# --- CONFIGURATION ---
basedir = os.path.abspath(os.path.dirname(__file__))

# Secret Key - Use environment variable or fallback to default (NOT SECURE FOR PRODUCTION!)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_super_secret_and_random_key_change_me')

# Database URI - Use environment variable or fallback to SQLite
database_url = os.getenv('DATABASE_URL')
if database_url:
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Fallback to SQLite for local development
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- EXTENSIONS ---
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# --- USER MAPPING (for sockets) ---
# A simple, in-memory dictionary to map user_id to their active socket (sid)
# { 1: "sid_of_user_1", 2: "sid_of_user_2" }
active_users = {}

# Track Google Meet acceptances per room
# Format: { "room_id": { "user_id_1": True, "user_id_2": False } }
google_meet_acceptances = {}

# Track Google Meet acceptances per room
# Format: { "room_id": { "user_id_1": True, "user_id_2": False } }
google_meet_acceptances = {}

# --- DATABASE MODEL ---
# Friends association table (many-to-many relationship)
friends_association = db.Table('friends',
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('friend_id', db.Integer, db.ForeignKey('user.id'), primary_key=True)
)

class FriendRequest(db.Model):
    """Model for friend requests"""
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, declined
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    
    # Relationships
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_requests')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_requests')
    
    def to_dict(self):
        return {
            'request_id': self.id,
            'sender_id': self.sender_id,
            'sender_name': self.sender.name,
            'sender_email': self.sender.email,
            'receiver_id': self.receiver_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    status = db.Column(db.String(50), nullable=False, default='idle')
    study_goal = db.Column(db.Text, nullable=True)
    goal_embedding = db.Column(PickleType, nullable=True)
    # Enhanced matching preferences
    preferred_session_length = db.Column(db.String(20), default='medium')  # short, medium, long
    study_style = db.Column(db.String(20), default='flexible')  # focused, collaborative, flexible
    timezone = db.Column(db.String(50), nullable=True)
    minimum_similarity_threshold = db.Column(db.Float, default=0.3)  # Minimum similarity score (0-1)
    last_match_time = db.Column(db.DateTime, nullable=True)  # Track last match to avoid spam
    # Trust bonus fields
    university = db.Column(db.String(200), nullable=True)  # University or school name
    location = db.Column(db.String(200), nullable=True)  # Location (city, state, country, etc.)
    
    # Gamification fields
    total_points = db.Column(db.Integer, default=0)  # Total XP/points earned
    current_streak = db.Column(db.Integer, default=0)  # Current consecutive days streak
    longest_streak = db.Column(db.Integer, default=0)  # Longest streak achieved
    last_study_date = db.Column(db.Date, nullable=True)  # Last date user studied
    achievements = db.Column(db.Text, nullable=True)  # JSON string of unlocked achievements
    
    # Friends relationship (many-to-many)
    friends = db.relationship('User',
        secondary=friends_association,
        primaryjoin=(friends_association.c.user_id == id),
        secondaryjoin=(friends_association.c.friend_id == id),
        backref=db.backref('friend_of', lazy='dynamic'),
        lazy='dynamic'
    )

    def to_dict(self):
        return {
            'user_id': self.id,
            'name': self.name,
            'email': self.email,
            'status': self.status
        }
        
    def to_match_profile(self):
        # A profile-safe version to send to other users
        return {
            'user_id': self.id,
            'name': self.name,
            'study_goal': self.study_goal
        }

class StudySession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    partner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    room_id = db.Column(db.String(200), nullable=True)
    study_goal = db.Column(db.Text, nullable=True)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.now)
    end_time = db.Column(db.DateTime, nullable=True)
    duration_seconds = db.Column(db.Integer, nullable=True)  # Duration in seconds
    pomodoro_count = db.Column(db.Integer, default=0)  # Number of pomodoros completed
    status = db.Column(db.String(50), default='active')  # active, completed, abandoned
    # Rating fields
    session_rating = db.Column(db.Integer, nullable=True)  # 1-5 rating for the session
    matchmaking_rating = db.Column(db.Integer, nullable=True)  # 1-5 rating for matchmaking algorithm
    feedback_text = db.Column(db.Text, nullable=True)  # Optional feedback text
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref='study_sessions')
    partner = db.relationship('User', foreign_keys=[partner_id])
    
    def to_dict(self):
        return {
            'session_id': self.id,
            'user_id': self.user_id,
            'partner_id': self.partner_id,
            'partner_name': self.partner.name if self.partner else None,
            'room_id': self.room_id,
            'study_goal': self.study_goal,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'duration_formatted': self.format_duration(),
            'pomodoro_count': self.pomodoro_count,
            'status': self.status,
            'session_rating': self.session_rating,
            'matchmaking_rating': self.matchmaking_rating,
            'feedback_text': self.feedback_text
        }
    
    def format_duration(self):
        if not self.duration_seconds:
            return "0:00"
        hours = self.duration_seconds // 3600
        minutes = (self.duration_seconds % 3600) // 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m"

# --- EMBEDDING HELPER ---
def get_embedding(text):
    try:
        result = genai.embed_content(model="models/embedding-001", content=text, task_type="RETRIEVAL_DOCUMENT")
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# --- AUTH DECORATOR ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['sub']).first()
        except Exception as e:
            return jsonify({'message': 'Token is invalid!', 'error': str(e)}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# --- ENHANCED MATCHING HELPERS ---

def calculate_historical_compatibility(user1_id, user2_id):
    """Calculate compatibility based on past study sessions together"""
    # Check for past successful sessions
    past_sessions = StudySession.query.filter(
        ((StudySession.user_id == user1_id) & (StudySession.partner_id == user2_id)) |
        ((StudySession.user_id == user2_id) & (StudySession.partner_id == user1_id)),
        StudySession.status == 'completed'
    ).all()
    
    if not past_sessions:
        return 0.5  # Neutral score if no history
    
    # Calculate success metrics
    total_sessions = len(past_sessions)
    avg_duration = sum(s.duration_seconds or 0 for s in past_sessions) / total_sessions if total_sessions > 0 else 0
    total_pomodoros = sum(s.pomodoro_count or 0 for s in past_sessions)
    
    # Score based on:
    # - Number of sessions (more = better, up to 0.3 points)
    # - Average duration (longer = better, up to 0.3 points)
    # - Pomodoro completion (more = better, up to 0.4 points)
    session_score = min(total_sessions * 0.1, 0.3)
    duration_score = min(avg_duration / 3600 * 0.3, 0.3)  # Normalize by hours
    pomodoro_score = min(total_pomodoros * 0.1, 0.4)
    
    historical_score = session_score + duration_score + pomodoro_score
    return min(historical_score, 1.0)  # Cap at 1.0

def calculate_preference_compatibility(user1, user2):
    """Calculate compatibility based on user preferences"""
    score = 0.0
    factors = 0
    
    # Session length preference (0.3 weight)
    if user1.preferred_session_length == user2.preferred_session_length:
        score += 0.3
    elif (user1.preferred_session_length == 'flexible' or 
          user2.preferred_session_length == 'flexible'):
        score += 0.15  # Partial match if one is flexible
    factors += 0.3
    
    # Study style compatibility (0.4 weight)
    style_compatibility = {
        ('focused', 'focused'): 0.4,
        ('collaborative', 'collaborative'): 0.4,
        ('flexible', 'flexible'): 0.3,
        ('focused', 'flexible'): 0.25,
        ('collaborative', 'flexible'): 0.35,
        ('focused', 'collaborative'): 0.1,  # Less compatible
    }
    style_key = tuple(sorted([user1.study_style, user2.study_style]))
    score += style_compatibility.get(style_key, 0.2)
    factors += 0.4
    
    # Timezone compatibility (0.3 weight) - if both have timezones
    if user1.timezone and user2.timezone:
        if user1.timezone == user2.timezone:
            score += 0.3
        else:
            # Partial score for different timezones (could be enhanced with actual time diff)
            score += 0.15
        factors += 0.3
    
    # Normalize score
    if factors > 0:
        return score / factors
    return 0.5  # Default neutral score

def calculate_trust_bonus(user1, user2):
    """Calculate trust bonus based on shared university, location, direct friends, or mutual friends"""
    bonus = 0.0
    reasons = []
    
    # Check if they are direct friends (highest bonus - 0.25)
    if user1.friends:
        user1_friend_ids = {f.id for f in user1.friends}
        if user2.id in user1_friend_ids:
            bonus += 0.25
            reasons.append("Direct friend")
    
    # Same university/school (0.15 bonus)
    if user1.university and user2.university:
        if user1.university.lower().strip() == user2.university.lower().strip():
            bonus += 0.15
            reasons.append("Same university/school")
    
    # Similar location (0.10 bonus)
    if user1.location and user2.location:
        location1 = user1.location.lower().strip()
        location2 = user2.location.lower().strip()
        # Exact match
        if location1 == location2:
            bonus += 0.10
            reasons.append("Same location")
        # Partial match (check if they share city or state)
        elif any(word in location2 for word in location1.split() if len(word) > 3):
            bonus += 0.05
            reasons.append("Similar location")
    
    # Mutual friends (0.20 bonus) - only if not already direct friends
    if bonus < 0.25 and user1.friends and user2.friends:
        user1_friend_ids = {f.id for f in user1.friends}
        user2_friend_ids = {f.id for f in user2.friends}
        mutual_friends = user1_friend_ids.intersection(user2_friend_ids)
        if mutual_friends:
            # Bonus increases with number of mutual friends (capped at 0.20)
            mutual_count = len(mutual_friends)
            mutual_bonus = min(0.20, 0.10 + (mutual_count - 1) * 0.05)
            bonus += mutual_bonus
            reasons.append(f"{mutual_count} mutual friend(s)")
    
    # Cap total bonus at 0.40 (40% boost) - increased to accommodate direct friend bonus
    bonus = min(bonus, 0.40)
    
    return bonus, reasons

def calculate_multi_factor_score(target_user, candidate, embedding_similarity, historical_score, preference_score):
    """Calculate final match score using multiple factors"""
    # Weighted combination:
    # - Embedding similarity: 35% (semantic match on goals)
    # - Historical compatibility: 20% (past success)
    # - Preference match: 15% (compatibility in style/preferences)
    # - Recency penalty: 10% (avoid matching same person too soon)
    # - Trust bonus: up to 20% (university, location, mutual friends)
    
    # Recency penalty - reduce score if matched recently
    recency_penalty = 1.0
    if target_user.last_match_time:
        time_since_last_match = (datetime.now() - target_user.last_match_time).total_seconds() / 3600
        if time_since_last_match < 1:  # Less than 1 hour
            recency_penalty = 0.5
        elif time_since_last_match < 24:  # Less than 24 hours
            recency_penalty = 0.7
        elif time_since_last_match < 168:  # Less than 1 week
            recency_penalty = 0.9
    
    # Calculate trust bonus
    trust_bonus, trust_reasons = calculate_trust_bonus(target_user, candidate)
    
    # Calculate base weighted score (normalized to 0.8 to leave room for trust bonus)
    base_score = (
        embedding_similarity * 0.35 +
        historical_score * 0.20 +
        preference_score * 0.15 +
        recency_penalty * 0.10
    )
    
    # Add trust bonus (can boost score up to 1.0)
    final_score = min(base_score + trust_bonus, 1.0)
    
    return final_score, trust_bonus, trust_reasons

def filter_and_rank_candidates(target_user, candidates, min_candidates=3, max_candidates=10):
    """Enhanced filtering and ranking of candidates"""
    if not candidates:
        return []
    
    # Filter out candidates below similarity threshold
    target_embedding = np.array(target_user.goal_embedding).reshape(1, -1)
    candidate_data = []
    
    for candidate in candidates:
        if candidate.goal_embedding is None:
            continue
        
        # Calculate embedding similarity
        candidate_embedding = np.array(candidate.goal_embedding).reshape(1, -1)
        similarity = cosine_similarity(target_embedding, candidate_embedding)[0][0]
        
        # Skip if below threshold
        threshold = max(target_user.minimum_similarity_threshold, 0.2)  # Minimum 0.2
        if similarity < threshold:
            continue
        
        # Calculate additional scores
        historical_score = calculate_historical_compatibility(target_user.id, candidate.id)
        preference_score = calculate_preference_compatibility(target_user, candidate)
        
        # Calculate final multi-factor score with trust bonus
        final_score, trust_bonus, trust_reasons = calculate_multi_factor_score(
            target_user, candidate, similarity, historical_score, preference_score
        )
        
        candidate_data.append({
            'user': candidate,
            'embedding_similarity': similarity,
            'historical_score': historical_score,
            'preference_score': preference_score,
            'trust_bonus': trust_bonus,
            'trust_reasons': trust_reasons,
            'final_score': final_score
        })
    
    # Sort by final score (descending)
    candidate_data.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Return top candidates
    return candidate_data[:max_candidates]
    
# --- LLM RERANKING HELPER ---
def rerank_candidates_with_llm(target_user, candidate_data_list):
    """Enhanced LLM reranking with detailed context"""
    candidates_text = ""
    for i, data in enumerate(candidate_data_list):
        candidate = data['user']
        trust_info = ""
        if data.get('trust_bonus', 0) > 0:
            trust_reasons = data.get('trust_reasons', [])
            trust_info = f"\n        - Trust Bonus: +{data['trust_bonus']:.2f} ({', '.join(trust_reasons)})"
        candidates_text += f"""
        Candidate {i+1}:
        - Name: {candidate.name}
        - Study Goal: "{candidate.study_goal}"
        - Study Style: {candidate.study_style}
        - Preferred Session Length: {candidate.preferred_session_length}
        - University: {candidate.university or 'Not specified'}
        - Location: {candidate.location or 'Not specified'}
        - Embedding Similarity: {data['embedding_similarity']:.2f}
        - Historical Compatibility: {data['historical_score']:.2f}
        - Preference Match: {data['preference_score']:.2f}{trust_info}
        - Overall Score: {data['final_score']:.2f}
        """
    
    prompt = f"""
    You are an advanced AI matchmaking system for study partners. Your goal is to find the best study match based on multiple factors.
    
    **Target User Profile:**
    - Name: "{target_user.name}"
    - Study Goal: "{target_user.study_goal}"
    - Study Style: {target_user.study_style}
    - Preferred Session Length: {target_user.preferred_session_length}
    
    **Potential Matches (pre-ranked by algorithm):**
    {candidates_text}
    
    **Your Task:**
    Analyze each candidate considering:
    1. Semantic similarity of study goals (how well do their goals align?)
    2. Study style compatibility (will they work well together?)
    3. Historical success (if they've studied together before, how did it go?)
    4. Overall fit for productive study sessions
    
    Return a ranked list in valid JSON format, from BEST to WORST match. For each candidate, provide:
    - user_id: The candidate's user ID
    - name: Candidate's name
    - compatibility_score: A score from 0.0 to 1.0 (higher is better)
    - justification: A brief explanation (2-3 sentences) of why this is a good/bad match
    
    Return ONLY a valid JSON array, no other text:
    [
      {{ "user_id": <id>, "name": "<name>", "compatibility_score": <0.0-1.0>, "justification": "<reason>" }}
    ]
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        # Try to extract JSON if wrapped in markdown
        if cleaned_response.startswith('['):
            ranked_list = json.loads(cleaned_response)
        else:
            # Try to find JSON array in the response
            start_idx = cleaned_response.find('[')
            end_idx = cleaned_response.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                ranked_list = json.loads(cleaned_response[start_idx:end_idx])
            else:
                raise ValueError("Could not parse JSON from response")
        
        # Map names to user IDs from candidate data
        candidate_map = {data['user'].name: data['user'].id for data in candidate_data_list}
        for item in ranked_list:
            if item.get("name") in candidate_map:
                item["user_id"] = candidate_map[item["name"]]
            # Also try to match by user_id if provided
            elif "user_id" not in item:
                # Fallback: use the first candidate's ID if name doesn't match
                for data in candidate_data_list:
                    if data['user'].name == item.get("name"):
                        item["user_id"] = data['user'].id
                        break
        
        return ranked_list
    except Exception as e:
        print(f"An error occurred during LLM re-ranking: {e}")
        print(f"Response was: {response.text if 'response' in locals() else 'No response'}")
        # Fallback: return candidates in original order with basic scores
        result = []
        for data in candidate_data_list:
            justification = f"Matched based on study goal similarity ({data['embedding_similarity']:.2f}) and preferences."
            if data.get('trust_bonus', 0) > 0:
                trust_reasons = data.get('trust_reasons', [])
                justification += f" Trust bonus: {', '.join(trust_reasons)}."
            result.append({
                "user_id": data['user'].id,
                "name": data['user'].name,
                "compatibility_score": round(data['final_score'], 2),
                "justification": justification
            })
        return result

# --- HTTP API ENDPOINTS ---

@app.route('/api/auth/register', methods=['POST'])
def register_user():
    data = request.get_json()
    user_exists = User.query.filter_by(email=data['email']).first()
    if user_exists:
        return jsonify({"error": "Email already exists"}), 409
    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(name=data['name'], email=data['email'], password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!", "user": new_user.to_dict()}), 201

@app.route('/api/auth/login', methods=['POST'])
def login_user():
    data = request.get_json()
    
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email and password are required"}), 400
    
    email = data['email'].strip()
    password = data['password']
    
    # Try to find user (case-insensitive email search)
    user = User.query.filter_by(email=email).first()
    
    if not user:
        # Try case-insensitive search
        all_users = User.query.all()
        user = next((u for u in all_users if u.email.lower() == email.lower()), None)
    
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Check password
    try:
        password_valid = bcrypt.check_password_hash(user.password_hash, password)
        if not password_valid:
            return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print(f"Error checking password: {e}")
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Generate token
    token = jwt.encode({
        'sub': str(user.id),
        'iat': datetime.now(timezone.utc),
        'exp': datetime.now(timezone.utc) + timedelta(hours=24)
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({"message": "Login successful!", "token": token})

@app.route('/api/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    """Get detailed user profile"""
    # Get user statistics
    completed_sessions = StudySession.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).count()
    
    total_study_time = sum(
        s.duration_seconds or 0 
        for s in StudySession.query.filter_by(
            user_id=current_user.id,
            status='completed'
        ).all()
    )
    
    return jsonify({
        "user_details": {
            "user_id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "status": current_user.status,
            "study_goal": current_user.study_goal,
            "preferred_session_length": current_user.preferred_session_length,
            "study_style": current_user.study_style,
            "timezone": current_user.timezone,
            "minimum_similarity_threshold": current_user.minimum_similarity_threshold,
            "university": current_user.university,
            "location": current_user.location,
            "stats": {
                "total_sessions": completed_sessions,
                "total_study_time_seconds": total_study_time,
                "total_study_time_formatted": format_duration(total_study_time)
            },
            "friends_count": current_user.friends.count() if current_user.friends else 0
        }
    })

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """Update user profile information"""
    data = request.get_json()
    
    if 'name' in data:
        current_user.name = data['name']
    
    if 'email' in data:
        # Check if email is already taken by another user
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user and existing_user.id != current_user.id:
            return jsonify({"error": "Email already in use"}), 400
        current_user.email = data['email']
    
    if 'preferred_session_length' in data:
        if data['preferred_session_length'] in ['short', 'medium', 'long']:
            current_user.preferred_session_length = data['preferred_session_length']
    
    if 'study_style' in data:
        if data['study_style'] in ['focused', 'collaborative', 'flexible']:
            current_user.study_style = data['study_style']
    
    if 'timezone' in data:
        current_user.timezone = data['timezone']
    
    if 'minimum_similarity_threshold' in data:
        threshold = float(data['minimum_similarity_threshold'])
        if 0.0 <= threshold <= 1.0:
            current_user.minimum_similarity_threshold = threshold
    
    if 'university' in data:
        current_user.university = data['university'].strip() if data['university'] else None
    
    if 'location' in data:
        current_user.location = data['location'].strip() if data['location'] else None
    
    db.session.commit()
    
    return jsonify({
        "message": "Profile updated successfully",
        "user_details": {
            "user_id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "preferred_session_length": current_user.preferred_session_length,
            "study_style": current_user.study_style,
            "timezone": current_user.timezone,
            "minimum_similarity_threshold": current_user.minimum_similarity_threshold,
            "university": current_user.university,
            "location": current_user.location
        }
    })

@app.route('/api/profile/change-password', methods=['POST'])
@token_required
def change_password(current_user):
    """Change user password"""
    data = request.get_json()
    
    if 'current_password' not in data or 'new_password' not in data:
        return jsonify({"error": "current_password and new_password are required"}), 400
    
    # Verify current password
    if not bcrypt.check_password_hash(current_user.password_hash, data['current_password']):
        return jsonify({"error": "Current password is incorrect"}), 401
    
    # Validate new password
    new_password = data['new_password']
    if len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters long"}), 400
    
    # Update password
    current_user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
    db.session.commit()
    
    return jsonify({"message": "Password changed successfully"})

@app.route('/api/friends', methods=['GET'])
@token_required
def get_friends(current_user):
    """Get list of user's friends"""
    friends = current_user.friends.all()
    return jsonify({
        "friends": [
            {
                "user_id": friend.id,
                "name": friend.name,
                "email": friend.email,
                "university": friend.university,
                "location": friend.location
            }
            for friend in friends
        ],
        "count": len(friends)
    })

@app.route('/api/friends/requests', methods=['GET'])
@token_required
def get_friend_requests(current_user):
    """Get pending friend requests (both sent and received)"""
    # Get received requests (pending)
    received_requests = FriendRequest.query.filter_by(
        receiver_id=current_user.id,
        status='pending'
    ).all()
    
    # Get sent requests (pending)
    sent_requests = FriendRequest.query.filter_by(
        sender_id=current_user.id,
        status='pending'
    ).all()
    
    # Build sent requests with receiver info
    sent_requests_data = []
    for req in sent_requests:
        receiver = User.query.get(req.receiver_id)
        req_dict = req.to_dict()
        if receiver:
            req_dict['receiver_name'] = receiver.name
            req_dict['receiver_email'] = receiver.email
        sent_requests_data.append(req_dict)
    
    return jsonify({
        "received_requests": [req.to_dict() for req in received_requests],
        "sent_requests": sent_requests_data,
        "received_count": len(received_requests),
        "sent_count": len(sent_requests)
    })

@app.route('/api/friends/send-request', methods=['POST'])
@token_required
def send_friend_request(current_user):
    """Send a friend request to another user"""
    data = request.get_json()
    friend_identifier = data.get('user_id') or data.get('email')
    
    if not friend_identifier:
        return jsonify({"error": "user_id or email is required"}), 400
    
    # Find the friend user
    if isinstance(friend_identifier, int) or (isinstance(friend_identifier, str) and friend_identifier.isdigit()):
        friend_user = User.query.get(int(friend_identifier))
    else:
        friend_user = User.query.filter_by(email=friend_identifier).first()
    
    if not friend_user:
        return jsonify({"error": "User not found"}), 404
    
    if friend_user.id == current_user.id:
        return jsonify({"error": "Cannot send friend request to yourself"}), 400
    
    # Check if already friends
    if current_user.friends.filter(friends_association.c.friend_id == friend_user.id).count() > 0:
        return jsonify({"error": "User is already your friend"}), 400
    
    # Check if there's already a pending request (either direction)
    existing_request = FriendRequest.query.filter(
        ((FriendRequest.sender_id == current_user.id) & (FriendRequest.receiver_id == friend_user.id)) |
        ((FriendRequest.sender_id == friend_user.id) & (FriendRequest.receiver_id == current_user.id)),
        FriendRequest.status == 'pending'
    ).first()
    
    if existing_request:
        if existing_request.sender_id == current_user.id:
            return jsonify({"error": "You have already sent a friend request to this user"}), 400
        else:
            return jsonify({"error": "This user has already sent you a friend request. Please accept it instead."}), 400
    
    # Create new friend request
    new_request = FriendRequest(
        sender_id=current_user.id,
        receiver_id=friend_user.id,
        status='pending'
    )
    db.session.add(new_request)
    db.session.commit()
    
    # Notify receiver via socket if they're online
    receiver_sid = active_users.get(str(friend_user.id))
    
    # Find if they're in a session together
    active_session = StudySession.query.filter_by(
        user_id=friend_user.id,
        status='active'
    ).first()
    
    session_room_id = None
    if active_session and active_session.partner_id == current_user.id:
        session_room_id = active_session.room_id
    
    friend_request_data = {
        'request_id': new_request.id,
        'sender_id': current_user.id,
        'sender_name': current_user.name,
        'sender_email': current_user.email,
        'session_room_id': session_room_id  # Include session room if they're in a session together
    }
    
    if receiver_sid:
        print(f"📤 Sending friend_request_received to receiver (sid: {receiver_sid})")
        socketio.emit('friend_request_received', friend_request_data, room=receiver_sid)
        # Also send to user room
        user_room = f"user_{friend_user.id}"
        print(f"📤 Sending friend_request_received to user room: {user_room}")
        socketio.emit('friend_request_received', friend_request_data, room=user_room)
        # If they're in a session together, also send to session room
        if session_room_id:
            print(f"📤 Sending friend_request_received to session room: {session_room_id}")
            socketio.emit('friend_request_received', friend_request_data, room=session_room_id)
            print(f"✓ Friend request notification sent to session room: {session_room_id}")
    else:
        print(f"⚠ Receiver {friend_user.id} is not online (no socket ID found)")
    
    return jsonify({
        "message": f"Friend request sent to {friend_user.name}",
        "request": new_request.to_dict()
    })

@app.route('/api/friends/accept-request', methods=['POST'])
@token_required
def accept_friend_request(current_user):
    """Accept a friend request"""
    data = request.get_json()
    request_id = data.get('request_id')
    
    if not request_id:
        return jsonify({"error": "request_id is required"}), 400
    
    # Find the request
    friend_request = FriendRequest.query.filter_by(
        id=request_id,
        receiver_id=current_user.id,
        status='pending'
    ).first()
    
    if not friend_request:
        return jsonify({"error": "Friend request not found or already processed"}), 404
    
    sender = User.query.get(friend_request.sender_id)
    if not sender:
        return jsonify({"error": "Sender not found"}), 404
    
    # Add both users as friends (bidirectional)
    current_user.friends.append(sender)
    sender.friends.append(current_user)
    
    # Mark request as accepted
    friend_request.status = 'accepted'
    db.session.commit()
    
    # Notify both users via socket if they're online
    sender_sid = active_users.get(str(sender.id))
    receiver_sid = active_users.get(str(current_user.id))
    
    # Find if they're in a session together
    sender_session = StudySession.query.filter_by(
        user_id=sender.id,
        status='active'
    ).first()
    receiver_session = StudySession.query.filter_by(
        user_id=current_user.id,
        status='active'
    ).first()
    
    session_room_id = None
    if sender_session and sender_session.partner_id == current_user.id:
        session_room_id = sender_session.room_id
    elif receiver_session and receiver_session.partner_id == sender.id:
        session_room_id = receiver_session.room_id
    
    # Prepare acceptance data
    sender_data = {
        'friend_id': current_user.id,
        'friend_name': current_user.name,
        'session_room_id': session_room_id
    }
    
    receiver_data = {
        'friend_id': sender.id,
        'friend_name': sender.name,
        'session_room_id': session_room_id
    }
    
    if sender_sid:
        socketio.emit('friend_request_accepted', sender_data, room=sender_sid)
        # Also send to user room
        socketio.emit('friend_request_accepted', sender_data, room=f"user_{sender.id}")
        # If they're in a session together, also send to session room
        if session_room_id:
            socketio.emit('friend_request_accepted', sender_data, room=session_room_id)
            print(f"✓ Friend request accepted notification sent to session room: {session_room_id}")
    
    if receiver_sid:
        socketio.emit('friend_request_accepted', receiver_data, room=receiver_sid)
        # Also send to user room
        socketio.emit('friend_request_accepted', receiver_data, room=f"user_{current_user.id}")
        # If they're in a session together, also send to session room
        if session_room_id:
            socketio.emit('friend_request_accepted', receiver_data, room=session_room_id)
            print(f"✓ Friend request accepted notification sent to session room: {session_room_id}")
    
    return jsonify({
        "message": f"Accepted friend request from {sender.name}",
        "friend": {
            "user_id": sender.id,
            "name": sender.name,
            "email": sender.email,
            "university": sender.university,
            "location": sender.location
        }
    })

@app.route('/api/friends/decline-request', methods=['POST'])
@token_required
def decline_friend_request(current_user):
    """Decline a friend request"""
    data = request.get_json()
    request_id = data.get('request_id')
    
    if not request_id:
        return jsonify({"error": "request_id is required"}), 400
    
    # Find the request
    friend_request = FriendRequest.query.filter_by(
        id=request_id,
        receiver_id=current_user.id,
        status='pending'
    ).first()
    
    if not friend_request:
        return jsonify({"error": "Friend request not found or already processed"}), 404
    
    # Mark request as declined
    friend_request.status = 'declined'
    db.session.commit()
    
    return jsonify({"message": "Friend request declined"})

@app.route('/api/friends/remove', methods=['POST'])
@token_required
def remove_friend(current_user):
    """Remove a friend"""
    data = request.get_json()
    friend_id = data.get('friend_id')
    
    if not friend_id:
        return jsonify({"error": "friend_id is required"}), 400
    
    friend_user = User.query.get(friend_id)
    if not friend_user:
        return jsonify({"error": "Friend not found"}), 404
    
    # Remove friend (bidirectional)
    if current_user.friends.filter(friends_association.c.friend_id == friend_id).count() > 0:
        current_user.friends.remove(friend_user)
        friend_user.friends.remove(current_user)
        db.session.commit()
        return jsonify({"message": f"Removed {friend_user.name} from friends"})
    else:
        return jsonify({"error": "User is not your friend"}), 400

@app.route('/api/users/me/status', methods=['PUT'])
@token_required
def update_status(current_user):
    data = request.get_json()
    new_status = data['status']
    if new_status not in ['idle', 'searching', 'in-session']:
        return jsonify({"error": "Invalid status value"}), 400
    
    if new_status == 'searching':
        if 'study_goal' not in data or not data['study_goal']:
            return jsonify({"error": "study_goal is required"}), 400
        current_user.study_goal = data['study_goal']
        embedding = get_embedding(current_user.study_goal)
        if embedding is None:
            return jsonify({"error": "Failed to generate embedding"}), 500
        current_user.goal_embedding = embedding
        
        # Update preferences if provided
        if 'preferred_session_length' in data:
            if data['preferred_session_length'] in ['short', 'medium', 'long']:
                current_user.preferred_session_length = data['preferred_session_length']
        if 'study_style' in data:
            if data['study_style'] in ['focused', 'collaborative', 'flexible']:
                current_user.study_style = data['study_style']
        if 'timezone' in data:
            current_user.timezone = data['timezone']
        if 'minimum_similarity_threshold' in data:
            threshold = float(data['minimum_similarity_threshold'])
            if 0.0 <= threshold <= 1.0:
                current_user.minimum_similarity_threshold = threshold
    else:
        current_user.study_goal = None
        current_user.goal_embedding = None
    
    current_user.status = new_status
    db.session.commit()
    
    # If setting status to searching, ensure socket is registered and user is in their room
    if new_status == 'searching':
        current_user_id_str = str(current_user.id)
        user_sid = active_users.get(current_user_id_str)
        if user_sid:
            # Ensure user is in their user room
            user_room = f"user_{current_user.id}"
            try:
                from flask_socketio import rooms as get_rooms
                user_rooms = get_rooms(user_sid)
                if user_room not in user_rooms:
                    print(f"⚠ User {current_user.id} not in their user room when setting status to searching")
                    print(f"⚠ Joining user {current_user.id} to room {user_room}...")
                    join_room(user_room, sid=user_sid)
                    print(f"✓ User {current_user.id} joined room {user_room}")
                    # Verify
                    user_rooms_after = get_rooms(user_sid)
                    print(f"✓ User {current_user.id} is now in rooms: {user_rooms_after}")
            except Exception as e:
                print(f"⚠ Could not ensure user is in room when setting status: {e}")
        else:
            print(f"⚠ User {current_user.id} not found in active_users when setting status to searching")
            print(f"⚠ Active users: {active_users}")
    
    return jsonify({"message": "Status updated", "user": current_user.to_dict()})

@app.route('/api/users/me/preferences', methods=['PUT'])
@token_required
def update_preferences(current_user):
    """Update user matching preferences"""
    data = request.get_json()
    
    if 'preferred_session_length' in data:
        if data['preferred_session_length'] in ['short', 'medium', 'long']:
            current_user.preferred_session_length = data['preferred_session_length']
    
    if 'study_style' in data:
        if data['study_style'] in ['focused', 'collaborative', 'flexible']:
            current_user.study_style = data['study_style']
    
    if 'timezone' in data:
        current_user.timezone = data['timezone']
    
    if 'minimum_similarity_threshold' in data:
        threshold = float(data['minimum_similarity_threshold'])
        if 0.0 <= threshold <= 1.0:
            current_user.minimum_similarity_threshold = threshold
    
    db.session.commit()
    return jsonify({
        "message": "Preferences updated",
        "preferences": {
            "preferred_session_length": current_user.preferred_session_length,
            "study_style": current_user.study_style,
            "timezone": current_user.timezone,
            "minimum_similarity_threshold": current_user.minimum_similarity_threshold
        }
    })

@app.route('/api/match/find', methods=['POST'])
@token_required
def find_match(current_user):
    """Enhanced matchmaking with multi-factor scoring"""
    # Check if current user is actually online
    current_user_id_str = str(current_user.id)
    if current_user_id_str not in active_users:
        return jsonify({"error": "You must be connected to search for matches. Please refresh the page."}), 400
    
    # Get all searching candidates
    candidates = User.query.filter(User.status == 'searching', User.id != current_user.id).all()
    if not candidates:
        return jsonify({"message": "No other users are currently searching."}), 404
    
    # Filter to only include users who are actually online (in active_users)
    online_candidate_ids = set()
    for user_id_str in active_users.keys():
        try:
            user_id = int(user_id_str)
            online_candidate_ids.add(user_id)
        except (ValueError, TypeError):
            continue
    
    # Filter candidates to only those who are online
    candidates = [c for c in candidates if c.id in online_candidate_ids]
    
    if not candidates:
        return jsonify({"message": "No other users are currently online and searching."}), 404
    
    if current_user.goal_embedding is None:
        return jsonify({"error": "You must set a study goal first."}), 400
    
    # Use enhanced filtering and ranking
    candidate_data = filter_and_rank_candidates(current_user, candidates, min_candidates=3, max_candidates=10)
    
    if not candidate_data:
        return jsonify({"message": "Could not find any suitable matches. Try adjusting your similarity threshold or study goal."}), 404
    
    # Get top candidates for LLM reranking (top 5 for better quality)
    top_candidates_for_llm = candidate_data[:5]
    
    # LLM reranking with enhanced context
    final_ranked_matches = rerank_candidates_with_llm(current_user, top_candidates_for_llm)
    
    if not final_ranked_matches:
        # Fallback: return algorithm-ranked matches
        final_ranked_matches = []
        for data in top_candidates_for_llm:
            justification = f"Matched based on study goal similarity ({data['embedding_similarity']:.2f}), historical compatibility ({data['historical_score']:.2f}), and preferences ({data['preference_score']:.2f})."
            if data.get('trust_bonus', 0) > 0:
                trust_reasons = data.get('trust_reasons', [])
                justification += f" Trust bonus: {', '.join(trust_reasons)}."
            final_ranked_matches.append({
                "user_id": data['user'].id,
                "name": data['user'].name,
                "compatibility_score": round(data['final_score'], 2),
                "justification": justification
            })
    
    # Update last_match_time to track recency
    current_user.last_match_time = datetime.now()
    db.session.commit()
    
    return jsonify(final_ranked_matches)

# --- NEW: INVITATION ENDPOINT ---
@app.route('/api/match/invite', methods=['POST'])
@token_required
def invite_user(current_user):
    data = request.get_json()
    if 'invitee_user_id' not in data:
        return jsonify({"error": "invitee_user_id is required"}), 400
    
    invitee_user_id = data['invitee_user_id']
    
    # Check if invitee user exists and is searching
    invitee_user = User.query.get(invitee_user_id)
    if not invitee_user:
        return jsonify({"error": "User not found."}), 404
    
    if invitee_user.status != 'searching':
        return jsonify({"error": f"User is not currently searching. Status: {invitee_user.status}"}), 400
    
    # Find the invitee's socket ID (sid) - try both string and int keys
    # Re-fetch right before sending to ensure we have the most current socket ID
    invitee_sid = active_users.get(str(invitee_user_id)) or active_users.get(invitee_user_id)
    
    if not invitee_sid:
        # Retry with a small delay - sometimes registration is still in progress
        import time
        for attempt in range(3):  # Try 3 times with 200ms delay
            time.sleep(0.2)
            invitee_sid = active_users.get(str(invitee_user_id)) or active_users.get(invitee_user_id)
            if invitee_sid:
                print(f"✓ Found invitee after {attempt + 1} attempt(s)")
                break
        
        if not invitee_sid:
            # Debug information
            print(f"✗ User {invitee_user_id} not found in active_users")
            print(f"Active users: {active_users}")
            print(f"Looking for user_id: {invitee_user_id} (as string: '{str(invitee_user_id)}')")
            return jsonify({
                "error": "User is not online or available. Please make sure they are logged in and have set their status to 'searching'.",
                "debug": {
                    "invitee_user_id": invitee_user_id,
                    "invitee_status": invitee_user.status,
                    "active_users_count": len(active_users),
                    "active_user_ids": list(active_users.keys())
                }
            }), 404
    
    # Re-fetch socket ID one more time right before sending (in case it changed)
    current_invitee_sid = active_users.get(str(invitee_user_id)) or active_users.get(invitee_user_id)
    if current_invitee_sid != invitee_sid:
        print(f"⚠ Socket ID changed from {invitee_sid} to {current_invitee_sid} - using updated ID")
        invitee_sid = current_invitee_sid
    
    # Prepare invite data
    invite_data = {
        'inviter': current_user.to_match_profile()
    }
    
    print(f"Attempting to send invitation:")
    print(f"  - From user {current_user.id} ({current_user.name})")
    print(f"  - To user {invitee_user_id} ({invitee_user.name})")
    print(f"  - Socket ID: {invitee_sid}")
    print(f"  - Invite data: {invite_data}")
    
    # Emit a real-time event *only* to the invited user
    # Use multiple methods to ensure delivery: socket ID, user room, and broadcast
    room_id = f"user_{invitee_user_id}"
    invite_sent = False
    
    try:
        # Method 1: Send directly to socket ID (most reliable if socket is connected)
        if invitee_sid:
            # Verify socket ID is in active_users (double-check)
            if invitee_sid in active_users.values():
                print(f"📤 Emitting 'invite_received' to socket {invitee_sid} with data: {invite_data}")
                socketio.emit('invite_received', invite_data, to=invitee_sid)
                print(f"✓ Invitation event emitted to socket {invitee_sid}")
                invite_sent = True
                
                # Also send a test event to verify connection (for debugging)
                socketio.emit('test_event', {'message': 'Connection test from invite', 'inviter_name': current_user.name}, to=invitee_sid)
                print(f"✓ Test event also sent to verify connection")
            else:
                print(f"⚠ Warning: Socket ID {invitee_sid} not found in active_users values")
                print(f"Active socket IDs: {list(active_users.values())}")
        
        # Method 2: Always send to user room as primary/fallback mechanism
        # Users join this room on registration, so this should always work
        print(f"📤 Broadcasting 'invite_received' to room {room_id} with data: {invite_data}")
        
        # Verify user is in the room before sending, and ensure they're joined
        from flask_socketio import rooms as get_rooms
        try:
            # Re-fetch current socket ID one more time (might have changed during processing)
            current_invitee_sid = active_users.get(str(invitee_user_id)) or active_users.get(invitee_user_id)
            
            if current_invitee_sid:
                user_rooms = get_rooms(current_invitee_sid)
                print(f"✓ Invitee (socket {current_invitee_sid}) is in rooms: {user_rooms}")
                if room_id not in user_rooms:
                    print(f"⚠ WARNING: Invitee not in their user room {room_id}!")
                    print(f"⚠ Attempting to join them to room {room_id}...")
                    try:
                        join_room(room_id, sid=current_invitee_sid)
                        print(f"✓ Joined invitee to room {room_id}")
                        # Verify again
                        user_rooms_after = get_rooms(current_invitee_sid)
                        print(f"✓ Invitee is now in rooms: {user_rooms_after}")
                        if room_id not in user_rooms_after:
                            print(f"⚠ WARNING: Still not in room after join attempt!")
                    except Exception as join_err:
                        print(f"✗ Could not join invitee to room: {join_err}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"✓ Invitee is already in room {room_id}")
            else:
                print(f"⚠ WARNING: Could not find current socket ID for invitee {invitee_user_id}")
                print(f"⚠ Active users: {active_users}")
        except Exception as e:
            print(f"⚠ Could not verify/join room membership: {e}")
            import traceback
            traceback.print_exc()
        
        # Send to user room (this should work even if user reconnected)
        socketio.emit('invite_received', invite_data, room=room_id)
        print(f"✓ Invitation broadcasted to room {room_id} (primary/fallback)")
        
        # Also send directly to current socket ID if we have it (as additional delivery method)
        current_invitee_sid = active_users.get(str(invitee_user_id)) or active_users.get(invitee_user_id)
        if current_invitee_sid:
            print(f"📤 Also sending directly to current socket ID {current_invitee_sid}")
            socketio.emit('invite_received', invite_data, to=current_invitee_sid)
            print(f"✓ Invitation also sent directly to socket {current_invitee_sid}")
        
        # Also send test_event to the room
        socketio.emit('test_event', {'message': 'Connection test from invite (room)', 'inviter_name': current_user.name}, room=room_id)
        print(f"✓ Test event also sent to room {room_id}")
        
        invite_sent = True
        
    except Exception as e:
        print(f"✗ Error emitting invitation: {e}")
        import traceback
        traceback.print_exc()
        # Still try to send via room even if socket ID method failed
        try:
            socketio.emit('invite_received', invite_data, room=room_id)
            print(f"✓ Fallback: Invitation sent via room {room_id} after error")
            invite_sent = True
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")
            return jsonify({"error": "Failed to send invitation event"}), 500
    
    if not invite_sent:
        return jsonify({"error": "Failed to send invitation - user may not be connected"}), 500
    
    return jsonify({
        "message": "Invitation sent successfully.",
        "debug": {
            "invitee_socket_id": invitee_sid,
            "invitee_user_id": invitee_user_id,
            "invitee_name": invitee_user.name
        }
    })

# --- STUDY SESSION & STATISTICS ENDPOINTS ---

@app.route('/api/sessions/start', methods=['POST'])
@token_required
def start_session(current_user):
    """Start tracking a new study session"""
    data = request.get_json()
    room_id = data.get('room_id')
    partner_id = data.get('partner_id')
    study_goal = data.get('study_goal', current_user.study_goal)
    
    # Check if user has an active session
    active_session = StudySession.query.filter_by(
        user_id=current_user.id,
        status='active'
    ).first()
    
    if active_session:
        return jsonify({"error": "You already have an active session", "session_id": active_session.id}), 400
    
    # Create new session
    new_session = StudySession(
        user_id=current_user.id,
        partner_id=partner_id,
        room_id=room_id,
        study_goal=study_goal,
        start_time=datetime.now(),
        status='active'
    )
    db.session.add(new_session)
    db.session.commit()
    
    return jsonify({"message": "Session started", "session": new_session.to_dict()}), 201

@app.route('/api/sessions/end', methods=['POST'])
@token_required
def end_session(current_user):
    """End the current active study session"""
    data = request.get_json()
    session_id = data.get('session_id')
    pomodoro_count = data.get('pomodoro_count', 0)
    
    # Find the session
    if session_id:
        session = StudySession.query.filter_by(id=session_id, user_id=current_user.id).first()
    else:
        # Find active session
        session = StudySession.query.filter_by(user_id=current_user.id, status='active').first()
    
    if not session:
        return jsonify({"error": "No active session found"}), 404
    
    # Calculate duration
    end_time = datetime.now()
    duration = int((end_time - session.start_time).total_seconds())
    
    session.end_time = end_time
    session.duration_seconds = duration
    session.pomodoro_count = pomodoro_count
    session.status = 'completed'
    
    # Update gamification stats
    today = datetime.now().date()
    last_study_date = current_user.last_study_date
    
    # Update streak
    if last_study_date:
        days_diff = (today - last_study_date).days
        if days_diff == 1:
            # Consecutive day - increment streak
            current_user.current_streak = (current_user.current_streak or 0) + 1
        elif days_diff > 1:
            # Streak broken - reset to 1
            current_user.current_streak = 1
        # If days_diff == 0, same day, don't change streak
    else:
        # First time studying
        current_user.current_streak = 1
    
    # Update longest streak
    if (current_user.current_streak or 0) > (current_user.longest_streak or 0):
        current_user.longest_streak = current_user.current_streak
    
    # Update last study date
    current_user.last_study_date = today
    
    # Award points: 10 points per pomodoro + 50 points per session
    points_earned = (pomodoro_count * 10) + 50
    current_user.total_points = (current_user.total_points or 0) + points_earned
    
    db.session.commit()
    
    return jsonify({
        "message": "Session ended", 
        "session": session.to_dict(),
        "points_earned": points_earned,
        "new_streak": current_user.current_streak
    })

@app.route('/api/sessions/rate', methods=['POST'])
@token_required
def rate_session(current_user):
    """Submit rating and feedback for a completed session"""
    data = request.get_json()
    session_id = data.get('session_id')
    room_id = data.get('room_id')  # Optional: can use room_id as fallback
    session_rating = data.get('session_rating')  # 1-5
    matchmaking_rating = data.get('matchmaking_rating')  # 1-5
    feedback_text = data.get('feedback_text', '')  # Optional text feedback
    
    if not session_id and not room_id:
        return jsonify({"error": "session_id or room_id is required"}), 400
    
    session = None
    
    # Method 1: Try to find by session_id and user_id first
    if session_id:
        session = StudySession.query.filter_by(id=session_id, user_id=current_user.id).first()
        if session:
            print(f"✓ Found session {session_id} for user {current_user.id}")
    
    # Method 2: If not found and room_id provided, try to find by room_id and user_id
    if not session and room_id:
        print(f"⚠ Session {session_id} not found, trying to find by room_id {room_id}...")
        session = StudySession.query.filter_by(
            user_id=current_user.id,
            room_id=room_id
        ).order_by(StudySession.end_time.desc(), StudySession.start_time.desc()).first()
        if session:
            print(f"✓ Found session {session.id} for user {current_user.id} by room_id {room_id}")
    
    # Method 3: Last resort - find most recent completed session for this user
    if not session:
        print(f"⚠ Session not found by session_id or room_id, trying most recent completed session...")
        session = StudySession.query.filter_by(
            user_id=current_user.id,
            status='completed'
        ).order_by(StudySession.end_time.desc()).first()
        if session:
            print(f"✓ Found most recent completed session {session.id} for user {current_user.id}")
    
    if not session:
        return jsonify({"error": "Session not found. Please make sure you have a completed session."}), 404
    
    # Allow rating for both active and completed sessions
    # (active sessions can be rated if user is leaving)
    if session.status not in ['active', 'completed']:
        return jsonify({"error": "Can only rate active or completed sessions"}), 400
    
    # If session is still active, mark it as completed first
    if session.status == 'active':
        if not session.end_time:
            session.end_time = datetime.now()
        if not session.duration_seconds:
            duration = int((session.end_time - session.start_time).total_seconds())
            session.duration_seconds = duration
        session.status = 'completed'
    
    # Validate ratings (1-5)
    if session_rating is not None:
        if not isinstance(session_rating, int) or session_rating < 1 or session_rating > 5:
            return jsonify({"error": "session_rating must be between 1 and 5"}), 400
        session.session_rating = session_rating
    
    if matchmaking_rating is not None:
        if not isinstance(matchmaking_rating, int) or matchmaking_rating < 1 or matchmaking_rating > 5:
            return jsonify({"error": "matchmaking_rating must be between 1 and 5"}), 400
        session.matchmaking_rating = matchmaking_rating
    
    if feedback_text:
        session.feedback_text = feedback_text[:1000]  # Limit to 1000 characters
    
    db.session.commit()
    
    return jsonify({
        "message": "Rating submitted successfully",
        "session": session.to_dict()
    })

@app.route('/api/stats', methods=['GET'])
@token_required
def get_statistics(current_user):
    """Get comprehensive study statistics for the user"""
    # Get all completed sessions
    completed_sessions = StudySession.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).all()
    
    # Helper function to get session duration (calculate if missing)
    def get_session_duration(session):
        if session.duration_seconds:
            return session.duration_seconds
        # Calculate from start_time to end_time if duration_seconds is missing
        if session.start_time and session.end_time:
            return int((session.end_time - session.start_time).total_seconds())
        return 0
    
    # Helper function to calculate study time from pomodoros
    # Each pomodoro is 25 minutes (1500 seconds) - standard pomodoro duration
    POMODORO_DURATION_SECONDS = 25 * 60  # 25 minutes in seconds
    
    def get_study_time(session):
        """Calculate actual study time from pomodoro count"""
        pomodoro_count = session.pomodoro_count or 0
        return pomodoro_count * POMODORO_DURATION_SECONDS
    
    # Calculate statistics
    total_sessions = len(completed_sessions)
    total_session_duration = sum(get_session_duration(s) for s in completed_sessions)  # Total session length
    total_study_time = sum(get_study_time(s) for s in completed_sessions)  # Total actual study time (timer running)
    total_pomodoros = sum(s.pomodoro_count or 0 for s in completed_sessions)
    
    # Average session duration and study time
    avg_session_duration = total_session_duration // total_sessions if total_sessions > 0 else 0
    avg_study_time = total_study_time // total_sessions if total_sessions > 0 else 0
    
    # Study goals breakdown
    goals_dict = {}
    for session in completed_sessions:
        goal = session.study_goal or "No goal"
        if goal not in goals_dict:
            goals_dict[goal] = {'count': 0, 'duration': 0}
        goals_dict[goal]['count'] += 1
        goals_dict[goal]['duration'] += get_session_duration(session)
    
    goals_breakdown = [
        {
            'goal': goal,
            'session_count': data['count'],
            'total_duration': data['duration'],
            'duration_formatted': format_duration(data['duration'])
        }
        for goal, data in goals_dict.items()
    ]
    
    # Today's statistics
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_sessions = StudySession.query.filter(
        StudySession.user_id == current_user.id,
        StudySession.status == 'completed',
        StudySession.end_time >= today_start
    ).all()
    
    today_session_duration = sum(get_session_duration(s) for s in today_sessions)
    today_study_time = sum(get_study_time(s) for s in today_sessions)
    today_pomodoros = sum(s.pomodoro_count or 0 for s in today_sessions)
    
    # Weekly statistics (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)
    weekly_sessions = StudySession.query.filter(
        StudySession.user_id == current_user.id,
        StudySession.status == 'completed',
        StudySession.end_time >= week_ago
    ).all()
    
    weekly_session_duration = sum(get_session_duration(s) for s in weekly_sessions)
    weekly_study_time = sum(get_study_time(s) for s in weekly_sessions)
    weekly_pomodoros = sum(s.pomodoro_count or 0 for s in weekly_sessions)
    
    # Monthly statistics (last 30 days)
    month_ago = datetime.now() - timedelta(days=30)
    monthly_sessions = StudySession.query.filter(
        StudySession.user_id == current_user.id,
        StudySession.status == 'completed',
        StudySession.end_time >= month_ago
    ).all()
    
    monthly_session_duration = sum(get_session_duration(s) for s in monthly_sessions)
    monthly_study_time = sum(get_study_time(s) for s in monthly_sessions)
    monthly_pomodoros = sum(s.pomodoro_count or 0 for s in monthly_sessions)
    
    # Daily breakdown for last 7 days
    daily_stats = []
    for i in range(7):
        day = datetime.now() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        day_sessions = StudySession.query.filter(
            StudySession.user_id == current_user.id,
            StudySession.status == 'completed',
            StudySession.end_time >= day_start,
            StudySession.end_time <= day_end
        ).all()
        
        day_session_duration = sum(get_session_duration(s) for s in day_sessions)
        day_study_time = sum(get_study_time(s) for s in day_sessions)
        daily_stats.append({
            'date': day_start.date().isoformat(),
            'session_duration': day_session_duration,
            'study_time': day_study_time,
            'sessions': len(day_sessions)
        })
    
    daily_stats.reverse()  # Oldest to newest
    
    return jsonify({
        'overall': {
            'total_sessions': total_sessions,
            'total_session_duration_seconds': total_session_duration,
            'total_session_duration_formatted': format_duration(total_session_duration),
            'total_study_time_seconds': total_study_time,
            'total_study_time_formatted': format_duration(total_study_time),
            'total_pomodoros': total_pomodoros,
            'avg_session_duration_seconds': avg_session_duration,
            'avg_session_duration_formatted': format_duration(avg_session_duration),
            'avg_study_time_seconds': avg_study_time,
            'avg_study_time_formatted': format_duration(avg_study_time)
        },
        'today': {
            'sessions': len(today_sessions),
            'session_duration_seconds': today_session_duration,
            'session_duration_formatted': format_duration(today_session_duration),
            'study_time_seconds': today_study_time,
            'study_time_formatted': format_duration(today_study_time),
            'pomodoros': today_pomodoros
        },
        'weekly': {
            'sessions': len(weekly_sessions),
            'session_duration_seconds': weekly_session_duration,
            'session_duration_formatted': format_duration(weekly_session_duration),
            'study_time_seconds': weekly_study_time,
            'study_time_formatted': format_duration(weekly_study_time),
            'pomodoros': weekly_pomodoros
        },
        'monthly': {
            'sessions': len(monthly_sessions),
            'session_duration_seconds': monthly_session_duration,
            'session_duration_formatted': format_duration(monthly_session_duration),
            'study_time_seconds': monthly_study_time,
            'study_time_formatted': format_duration(monthly_study_time),
            'pomodoros': monthly_pomodoros
        },
        'goals_breakdown': goals_breakdown,
        'daily_stats': daily_stats
    })

@app.route('/api/leaderboard/weekly', methods=['GET'])
@token_required
def get_weekly_leaderboard(current_user):
    """Get weekly leaderboard for study time"""
    week_ago = datetime.now() - timedelta(days=7)
    POMODORO_DURATION_SECONDS = 25 * 60
    
    # Get all users with completed sessions in the last week
    weekly_sessions = StudySession.query.filter(
        StudySession.status == 'completed',
        StudySession.end_time >= week_ago
    ).all()
    
    # Calculate study time for each user
    user_study_times = {}
    for session in weekly_sessions:
        user_id = session.user_id
        pomodoro_count = session.pomodoro_count or 0
        study_time = pomodoro_count * POMODORO_DURATION_SECONDS
        
        if user_id not in user_study_times:
            user_study_times[user_id] = {
                'user_id': user_id,
                'study_time': 0,
                'sessions': 0,
                'pomodoros': 0
            }
        
        user_study_times[user_id]['study_time'] += study_time
        user_study_times[user_id]['sessions'] += 1
        user_study_times[user_id]['pomodoros'] += pomodoro_count
    
    # Get user details and format leaderboard
    leaderboard = []
    for user_id, data in user_study_times.items():
        user = User.query.get(user_id)
        if user:
            leaderboard.append({
                'user_id': user_id,
                'name': user.name,
                'study_time_seconds': data['study_time'],
                'study_time_formatted': format_duration(data['study_time']),
                'sessions': data['sessions'],
                'pomodoros': data['pomodoros'],
                'is_current_user': user_id == current_user.id
            })
    
    # Sort by study time (descending)
    leaderboard.sort(key=lambda x: x['study_time_seconds'], reverse=True)
    
    # Add rank
    for i, entry in enumerate(leaderboard, 1):
        entry['rank'] = i
    
    # Get current user's rank if not in top list
    current_user_rank = None
    for entry in leaderboard:
        if entry['is_current_user']:
            current_user_rank = entry['rank']
            break
    
    return jsonify({
        'leaderboard': leaderboard[:50],  # Top 50
        'current_user_rank': current_user_rank,
        'period': 'weekly'
    })

@app.route('/api/leaderboard/monthly', methods=['GET'])
@token_required
def get_monthly_leaderboard(current_user):
    """Get monthly leaderboard for study time"""
    month_ago = datetime.now() - timedelta(days=30)
    POMODORO_DURATION_SECONDS = 25 * 60
    
    # Get all users with completed sessions in the last month
    monthly_sessions = StudySession.query.filter(
        StudySession.status == 'completed',
        StudySession.end_time >= month_ago
    ).all()
    
    # Calculate study time for each user
    user_study_times = {}
    for session in monthly_sessions:
        user_id = session.user_id
        pomodoro_count = session.pomodoro_count or 0
        study_time = pomodoro_count * POMODORO_DURATION_SECONDS
        
        if user_id not in user_study_times:
            user_study_times[user_id] = {
                'user_id': user_id,
                'study_time': 0,
                'sessions': 0,
                'pomodoros': 0
            }
        
        user_study_times[user_id]['study_time'] += study_time
        user_study_times[user_id]['sessions'] += 1
        user_study_times[user_id]['pomodoros'] += pomodoro_count
    
    # Get user details and format leaderboard
    leaderboard = []
    for user_id, data in user_study_times.items():
        user = User.query.get(user_id)
        if user:
            leaderboard.append({
                'user_id': user_id,
                'name': user.name,
                'study_time_seconds': data['study_time'],
                'study_time_formatted': format_duration(data['study_time']),
                'sessions': data['sessions'],
                'pomodoros': data['pomodoros'],
                'is_current_user': user_id == current_user.id
            })
    
    # Sort by study time (descending)
    leaderboard.sort(key=lambda x: x['study_time_seconds'], reverse=True)
    
    # Add rank
    for i, entry in enumerate(leaderboard, 1):
        entry['rank'] = i
    
    # Get current user's rank
    current_user_rank = None
    for entry in leaderboard:
        if entry['is_current_user']:
            current_user_rank = entry['rank']
            break
    
    return jsonify({
        'leaderboard': leaderboard[:50],  # Top 50
        'current_user_rank': current_user_rank,
        'period': 'monthly'
    })

@app.route('/api/gamification/stats', methods=['GET'])
@token_required
def get_gamification_stats(current_user):
    """Get gamification statistics for current user"""
    # Calculate current streak
    today = datetime.now().date()
    current_streak = current_user.current_streak or 0
    longest_streak = current_user.longest_streak or 0
    total_points = current_user.total_points or 0
    
    # Get achievements (parse JSON if exists)
    achievements = []
    if current_user.achievements:
        try:
            import json
            achievements = json.loads(current_user.achievements)
        except:
            achievements = []
    
    # Calculate achievements based on stats
    completed_sessions = StudySession.query.filter_by(
        user_id=current_user.id,
        status='completed'
    ).all()
    
    POMODORO_DURATION_SECONDS = 25 * 60
    total_study_time = sum((s.pomodoro_count or 0) * POMODORO_DURATION_SECONDS for s in completed_sessions)
    total_pomodoros = sum(s.pomodoro_count or 0 for s in completed_sessions)
    
    # Check for new achievements
    new_achievements = []
    
    # First Session
    if len(completed_sessions) >= 1 and 'first_session' not in achievements:
        new_achievements.append({
            'id': 'first_session',
            'name': 'Getting Started',
            'description': 'Complete your first study session',
            'icon': '🎯'
        })
        achievements.append('first_session')
    
    # 10 Sessions
    if len(completed_sessions) >= 10 and 'ten_sessions' not in achievements:
        new_achievements.append({
            'id': 'ten_sessions',
            'name': 'Dedicated Learner',
            'description': 'Complete 10 study sessions',
            'icon': '📚'
        })
        achievements.append('ten_sessions')
    
    # 50 Sessions
    if len(completed_sessions) >= 50 and 'fifty_sessions' not in achievements:
        new_achievements.append({
            'id': 'fifty_sessions',
            'name': 'Study Master',
            'description': 'Complete 50 study sessions',
            'icon': '🏆'
        })
        achievements.append('fifty_sessions')
    
    # 100 Pomodoros
    if total_pomodoros >= 100 and 'hundred_pomodoros' not in achievements:
        new_achievements.append({
            'id': 'hundred_pomodoros',
            'name': 'Pomodoro Pro',
            'description': 'Complete 100 pomodoros',
            'icon': '🍅'
        })
        achievements.append('hundred_pomodoros')
    
    # 10 Hour Study Time
    if total_study_time >= 10 * 3600 and 'ten_hours' not in achievements:
        new_achievements.append({
            'id': 'ten_hours',
            'name': '10 Hour Club',
            'description': 'Study for 10 hours total',
            'icon': '⏰'
        })
        achievements.append('ten_hours')
    
    # 7 Day Streak
    if current_streak >= 7 and 'seven_day_streak' not in achievements:
        new_achievements.append({
            'id': 'seven_day_streak',
            'name': 'Week Warrior',
            'description': 'Maintain a 7-day study streak',
            'icon': '🔥'
        })
        achievements.append('seven_day_streak')
    
    # 30 Day Streak
    if current_streak >= 30 and 'thirty_day_streak' not in achievements:
        new_achievements.append({
            'id': 'thirty_day_streak',
            'name': 'Monthly Champion',
            'description': 'Maintain a 30-day study streak',
            'icon': '💪'
        })
        achievements.append('thirty_day_streak')
    
    # Update achievements if new ones were unlocked
    if new_achievements:
        import json
        current_user.achievements = json.dumps(achievements)
        # Award points for new achievements
        current_user.total_points = (current_user.total_points or 0) + len(new_achievements) * 100
        db.session.commit()
    
    # Format existing achievements
    achievement_list = []
    achievement_map = {
        'first_session': {'name': 'Getting Started', 'description': 'Complete your first study session', 'icon': '🎯'},
        'ten_sessions': {'name': 'Dedicated Learner', 'description': 'Complete 10 study sessions', 'icon': '📚'},
        'fifty_sessions': {'name': 'Study Master', 'description': 'Complete 50 study sessions', 'icon': '🏆'},
        'hundred_pomodoros': {'name': 'Pomodoro Pro', 'description': 'Complete 100 pomodoros', 'icon': '🍅'},
        'ten_hours': {'name': '10 Hour Club', 'description': 'Study for 10 hours total', 'icon': '⏰'},
        'seven_day_streak': {'name': 'Week Warrior', 'description': 'Maintain a 7-day study streak', 'icon': '🔥'},
        'thirty_day_streak': {'name': 'Monthly Champion', 'description': 'Maintain a 30-day study streak', 'icon': '💪'}
    }
    
    for ach_id in achievements:
        if ach_id in achievement_map:
            achievement_list.append({
                'id': ach_id,
                **achievement_map[ach_id]
            })
    
    return jsonify({
        'points': total_points,
        'current_streak': current_streak,
        'longest_streak': longest_streak,
        'achievements': achievement_list,
        'new_achievements': new_achievements
    })

@app.route('/api/sessions/partner', methods=['GET'])
@token_required
def get_session_partner(current_user):
    """Get partner information for current active session"""
    room_id = request.args.get('room_id')
    if not room_id:
        return jsonify({"error": "room_id is required"}), 400
    
    # Find active session for this user in this room
    print(f"🔍 Looking for session - user_id: {current_user.id}, room_id: {room_id}")
    session = StudySession.query.filter_by(
        user_id=current_user.id,
        room_id=room_id,
        status='active'
    ).first()
    
    if not session:
        # Try to find any active session for this user to help debug
        any_session = StudySession.query.filter_by(
            user_id=current_user.id,
            status='active'
        ).first()
        if any_session:
            print(f"⚠ Found active session but with different room_id: {any_session.room_id} (looking for: {room_id})")
            # If we found a session with a different room_id, use it (might be a timing issue or room_id mismatch)
            # But only if it has a partner_id
            if any_session.partner_id:
                session = any_session
                print(f"✓ Using session with room_id: {session.room_id} (requested: {room_id})")
            else:
                print(f"⚠ Active session found but no partner_id, cannot use it")
                return jsonify({"error": "No active session or partner found"}), 404
        else:
            # Try to find a recently completed session (within last 5 minutes) as fallback
            # This handles cases where session was marked completed due to brief disconnect
            from datetime import timedelta
            recent_time = datetime.now() - timedelta(minutes=5)
            recent_session = StudySession.query.filter(
                StudySession.user_id == current_user.id,
                StudySession.room_id == room_id,
                StudySession.status == 'completed',
                StudySession.end_time >= recent_time
            ).order_by(StudySession.end_time.desc()).first()
            
            if recent_session and recent_session.partner_id:
                print(f"⚠ No active session, but found recently completed session (ended {recent_session.end_time})")
                print(f"✓ Using recently completed session as fallback - will reactivate it")
                # Reactivate the session so it can be used
                recent_session.status = 'active'
                db.session.commit()
                session = recent_session
            else:
                print(f"⚠ No active session found for user {current_user.id}")
                # List all sessions for this user (for debugging)
                all_sessions = StudySession.query.filter_by(user_id=current_user.id).order_by(StudySession.start_time.desc()).limit(5).all()
                print(f"📋 Recent sessions for user {current_user.id}:")
                for s in all_sessions:
                    print(f"  - Session {s.id}: room_id={s.room_id}, status={s.status}, partner_id={s.partner_id}, end_time={s.end_time}")
                return jsonify({"error": "No active session or partner found"}), 404
    
    if not session.partner_id:
        print(f"⚠ Session found but no partner_id: session_id={session.id}")
        return jsonify({"error": "No active session or partner found"}), 404
    
    print(f"✓ Session found: session_id={session.id}, partner_id={session.partner_id}")
    
    partner = User.query.get(session.partner_id)
    if not partner:
        return jsonify({"error": "Partner not found"}), 404
    
    # Check if they are friends
    are_friends = current_user.friends.filter_by(id=partner.id).first() is not None
    
    # Check if partner is online
    partner_online = str(partner.id) in active_users
    
    # Check for existing friend requests (either direction)
    friend_request_status = None
    friend_request_id = None
    
    if not are_friends:
        # Check if current user sent a request to partner
        sent_request = FriendRequest.query.filter_by(
            sender_id=current_user.id,
            receiver_id=partner.id,
            status='pending'
        ).first()
        
        if sent_request:
            friend_request_status = 'sent'
            friend_request_id = sent_request.id
        else:
            # Check if partner sent a request to current user
            received_request = FriendRequest.query.filter_by(
                sender_id=partner.id,
                receiver_id=current_user.id,
                status='pending'
            ).first()
            
            if received_request:
                friend_request_status = 'received'
                friend_request_id = received_request.id
    
    return jsonify({
        "partner": {
            "id": partner.id,
            "name": partner.name,
            "email": partner.email,  # Always include email (needed for friend request even if not friends)
            "university": partner.university if are_friends else None,
            "location": partner.location if are_friends else None
        },
        "are_friends": are_friends,
        "partner_online": partner_online,
        "friend_request_status": friend_request_status,  # 'sent', 'received', or None
        "friend_request_id": friend_request_id
    })

@app.route('/api/users/online', methods=['GET'])
@token_required
def check_user_online(current_user):
    """Check if a specific user is online"""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400
    
    try:
        user_id_int = int(user_id)
    except:
        return jsonify({"error": "Invalid user_id"}), 400
    
    is_online = str(user_id_int) in active_users
    return jsonify({
        "user_id": user_id_int,
        "is_online": is_online
    })

@app.route('/api/match/invite-friend', methods=['POST'])
@token_required
def invite_friend_to_session(current_user):
    """Invite a friend directly to a session"""
    data = request.get_json()
    friend_id = data.get('friend_id')
    room_id = data.get('room_id')
    
    if not friend_id or not room_id:
        return jsonify({"error": "friend_id and room_id are required"}), 400
    
    try:
        friend_id_int = int(friend_id)
    except:
        return jsonify({"error": "Invalid friend_id"}), 400
    
    # Check if they are friends
    friend = current_user.friends.filter_by(id=friend_id_int).first()
    if not friend:
        return jsonify({"error": "User is not your friend"}), 403
    
    # Check if friend is online
    friend_sid = active_users.get(str(friend_id_int))
    if not friend_sid:
        return jsonify({"error": "Friend is not online"}), 404
    
    # Send invitation
    socketio.emit('invite_received', {
        'inviter_id': current_user.id,
        'inviter_name': current_user.name,
        'room_id': room_id,
        'study_goal': current_user.study_goal
    }, room=friend_sid)
    
    # Also send to user room as fallback
    friend_room = f"user_{friend_id_int}"
    socketio.emit('invite_received', {
        'inviter_id': current_user.id,
        'inviter_name': current_user.name,
        'room_id': room_id,
        'study_goal': current_user.study_goal
    }, room=friend_room)
    
    return jsonify({
        "message": "Invitation sent to friend",
        "friend_name": friend.name
    })

@app.route('/api/sessions/history', methods=['GET'])
@token_required
def get_session_history(current_user):
    """Get session history with pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    limit = request.args.get('limit', None, type=int)
    
    query = StudySession.query.filter_by(user_id=current_user.id)
    
    if limit:
        sessions = query.order_by(StudySession.start_time.desc()).limit(limit).all()
        total = len(sessions)
    else:
        pagination = query.order_by(StudySession.start_time.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        sessions = pagination.items
        total = pagination.total
    
    return jsonify({
        'sessions': [s.to_dict() for s in sessions],
        'total': total,
        'page': page if not limit else 1,
        'per_page': per_page if not limit else total
    })

def format_duration(seconds):
    """Helper function to format duration in seconds to human-readable format"""
    if not seconds:
        return "0m"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

# --- SOCKET.IO EVENT HANDLERS ---

@socketio.on('timer_action')
def handle_timer_action(data):
    room_id = data.get('room_id')
    action = data.get('action')
    time = data.get('time') # We get the time from the client that set it
    mode = data.get('mode', 'pomodoro') # Get mode if provided
    
    user_id = None
    for uid, sid in active_users.items():
        if sid == request.sid:
            user_id = uid
            break
            
    if user_id and room_id:
        # Broadcast the action to everyone in the room
        # For reset action, ensure we send the exact time
        update_data = {
            'action': action,
            'mode': mode  # Include mode in update
        }
        
        # Always include time for reset action to ensure both users reset to same value
        if action == 'reset' and time is not None:
            update_data['time'] = time
        elif time is not None:
            update_data['time'] = time
        
        socketio.emit('timer_update', update_data, room=room_id)
        print(f"✓ Timer action '{action}' broadcasted to room {room_id} (time: {update_data.get('time', 'N/A')}, mode: {mode})")

@socketio.on('pomodoro_completed')
def handle_pomodoro_completed(data):
    """Track when a pomodoro is completed"""
    room_id = data.get('room_id')
    
    user_id = None
    for uid, sid in active_users.items():
        if sid == request.sid:
            user_id = uid
            break
    
    if user_id and room_id:
        # Update the active session's pomodoro count
        session = StudySession.query.filter_by(
            user_id=user_id,
            room_id=room_id,
            status='active'
        ).first()
        
        if session:
            session.pomodoro_count = (session.pomodoro_count or 0) + 1
            db.session.commit()


@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('register_connection')
def handle_register_connection(data):
    token = data.get('token')
    if not token:
        print("No token provided, connection not registered")
        socketio.emit('registration_failed', {'error': 'No token provided'}, room=request.sid)
        return
    try:
        # Decode token to get user_id
        token_data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
        user_id = token_data['sub']
        
        # Map the user_id to their current socket ID
        active_users[str(user_id)] = request.sid
        print(f"✓ Registered user {user_id} to sid {request.sid}")
        print(f"Active users: {active_users}")
        
        # Also join a user-specific room for fallback message delivery
        room_id = f"user_{user_id}"
        try:
            join_room(room_id, sid=request.sid)
            print(f"✓ User {user_id} joined room {room_id} for fallback messaging")
            
            # Verify room membership
            from flask_socketio import rooms as get_rooms
            user_rooms = get_rooms(request.sid)
            print(f"✓ User {user_id} is now in rooms: {user_rooms}")
            if room_id not in user_rooms:
                print(f"⚠ WARNING: User {user_id} not in their user room {room_id} after join attempt!")
        except Exception as e:
            print(f"✗ Error joining user {user_id} to room {room_id}: {e}")
            import traceback
            traceback.print_exc()
        
        # Send confirmation back to client
        socketio.emit('registration_confirmed', {
            'user_id': user_id,
            'status': 'registered'
        }, room=request.sid)
        
    except Exception as e:
        print(f"✗ Token invalid, connection not registered: {e}")
        socketio.emit('registration_failed', {'error': str(e)}, room=request.sid)

@socketio.on('decline_invitation')
def handle_decline_invitation(data):
    """Handle when a user declines an invitation"""
    inviter_user_id = data.get('inviter_user_id')
    
    if not inviter_user_id:
        return
    
    print(f"❌ Invitation declined - Inviter: {inviter_user_id}")
    
    # Find the inviter's socket ID
    inviter_user_id_str = str(inviter_user_id)
    inviter_sid = active_users.get(inviter_user_id_str) or active_users.get(inviter_user_id)
    
    if inviter_sid:
        # Notify the inviter that their invite was declined
        socketio.emit('invite_declined', {
            'message': 'Your invitation was declined.'
        }, to=inviter_sid)
        print(f"✓ Notified inviter {inviter_user_id} that invite was declined")
    else:
        print(f"⚠ Could not find inviter {inviter_user_id} to notify about decline")

@socketio.on('accept_invitation')
def handle_accept_invitation(data):
    print(f"🎯 ACCEPT INVITATION EVENT RECEIVED")
    print(f"  - Socket ID: {request.sid}")
    print(f"  - Data received: {data}")
    
    inviter_user_id = data.get('inviter_user_id')
    
    if not inviter_user_id:
        error_msg = "ERROR: No inviter_user_id provided in accept_invitation"
        print(error_msg)
        try:
            socketio.emit('error', {'message': 'Invalid invitation data'}, to=request.sid)
        except:
            socketio.emit('error', {'message': 'Invalid invitation data'}, room=request.sid)
        return
    
    # Convert to string for consistency
    inviter_user_id_str = str(inviter_user_id)
    
    print(f"✓ Processing accept invitation:")
    print(f"  - Inviter user ID: {inviter_user_id} (as string: {inviter_user_id_str})")
    print(f"  - Socket ID: {request.sid}")
    print(f"  - Active users: {active_users}")
    
    # Find the inviter's socket ID (try both string and int keys)
    inviter_sid = active_users.get(inviter_user_id_str) or active_users.get(inviter_user_id)
    
    # Find the invitee (current user) by socket ID
    invitee_user_id = None
    invitee_user = None
    for user_id, sid in active_users.items():
        if sid == request.sid:
            invitee_user_id = user_id
            # Try to get user by string or int
            invitee_user = User.query.get(int(user_id))
            break
    
    print(f"Inviter SID: {inviter_sid}, Invitee user_id: {invitee_user_id}")
    
    if not inviter_sid:
        error_msg = f"Inviter (user {inviter_user_id}) is not online"
        print(f"ERROR: {error_msg}")
        try:
            socketio.emit('error', {'message': error_msg}, to=request.sid)
        except:
            socketio.emit('error', {'message': error_msg}, room=request.sid)
        return
    
    if not invitee_user:
        error_msg = f"Could not find invitee user for socket {request.sid}"
        print(f"ERROR: {error_msg}")
        try:
            socketio.emit('error', {'message': 'Could not identify your account'}, to=request.sid)
        except:
            socketio.emit('error', {'message': 'Could not identify your account'}, room=request.sid)
        return
    
    # Get inviter user (handle both int and string)
    try:
        inviter_user_id_int = int(inviter_user_id)
    except (ValueError, TypeError):
        inviter_user_id_int = inviter_user_id  # Keep as is if conversion fails
    
    inviter_user = User.query.get(inviter_user_id_int)
    
    if not inviter_user:
        error_msg = f"Inviter user {inviter_user_id} not found in database"
        print(f"ERROR: {error_msg}")
        try:
            socketio.emit('error', {'message': 'Inviter not found'}, to=request.sid)
        except:
            socketio.emit('error', {'message': 'Inviter not found'}, room=request.sid)
        return
    
    print(f"✓ Both users found: inviter={inviter_user.name} (id: {inviter_user_id_int}), invitee={invitee_user.name} (id: {invitee_user.id})")
        
    room_id = f"room_{inviter_user_id_int}_{invitee_user.id}"
    
    # --- Make both users join the Socket.IO room FIRST ---
    # This ensures they're in the room before we send events
    # Get current inviter SID (might have changed)
    current_inviter_sid = active_users.get(str(inviter_user_id_int)) or inviter_sid
    final_inviter_sid = current_inviter_sid
    
    print(f"📋 Room joining - Inviter SID: {final_inviter_sid}, Invitee SID: {request.sid}")
    
    try:
        join_room(room_id, sid=final_inviter_sid)
        print(f"✓ Inviter joined room {room_id} (sid: {final_inviter_sid})")
    except Exception as e:
        print(f"✗ Error joining inviter to room: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        join_room(room_id, sid=request.sid)
        print(f"✓ Invitee joined room {room_id} (sid: {request.sid})")
    except Exception as e:
        print(f"✗ Error joining invitee to room: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify both are in the room (for debugging)
    from flask_socketio import rooms as get_rooms
    try:
        inviter_rooms = get_rooms(final_inviter_sid)
        invitee_rooms = get_rooms(request.sid)
        print(f"✓ Room membership verified - Inviter rooms: {inviter_rooms}, Invitee rooms: {invitee_rooms}")
        if room_id not in inviter_rooms:
            print(f"⚠ WARNING: Inviter not in room {room_id}!")
        if room_id not in invitee_rooms:
            print(f"⚠ WARNING: Invitee not in room {room_id}!")
    except Exception as e:
        print(f"✗ Error checking room membership: {e}")
    
    # --- Start tracking study sessions for both users ---
    try:
        # Check if sessions already exist
        # Use inviter_user_id_int for consistency (sessions are created with int)
        inviter_session = StudySession.query.filter_by(
            user_id=inviter_user_id_int,
            room_id=room_id,
            status='active'
        ).first()
        
        invitee_session = StudySession.query.filter_by(
            user_id=invitee_user.id,
            room_id=room_id,
            status='active'
        ).first()
        
        # Create session for inviter if doesn't exist
        if not inviter_session:
            inviter_session = StudySession(
                user_id=inviter_user_id_int,
                partner_id=invitee_user.id,
                room_id=room_id,
                study_goal=inviter_user.study_goal,
                start_time=datetime.now(),
                status='active'
            )
            db.session.add(inviter_session)
        
        # Create session for invitee if doesn't exist
        if not invitee_session:
            invitee_session = StudySession(
                user_id=invitee_user.id,
                partner_id=inviter_user_id_int,
                room_id=room_id,
                study_goal=invitee_user.study_goal,
                start_time=datetime.now(),
                status='active'
            )
            db.session.add(invitee_session)
        
        db.session.commit()
        
        # Refresh sessions from database to ensure they're fully committed
        db.session.refresh(inviter_session)
        db.session.refresh(invitee_session)
        
        print(f"✓ Sessions created and committed:")
        print(f"  - Inviter: session_id={inviter_session.id}, user_id={inviter_session.user_id}, room_id={inviter_session.room_id}, partner_id={inviter_session.partner_id}")
        print(f"  - Invitee: session_id={invitee_session.id}, user_id={invitee_session.user_id}, room_id={invitee_session.room_id}, partner_id={invitee_session.partner_id}")
        
        # Re-lookup inviter's socket ID right before sending (in case it changed)
        current_inviter_sid = active_users.get(str(inviter_user_id_int)) or active_users.get(inviter_user_id_int)
        if current_inviter_sid != final_inviter_sid:
            print(f"⚠ Inviter socket ID changed: {final_inviter_sid} -> {current_inviter_sid}")
            final_inviter_sid = current_inviter_sid
        
        print(f"✓ Sending match_confirmed to inviter (sid: {final_inviter_sid}) and invitee (sid: {request.sid})")
        print(f"✓ Active users at send time: {active_users}")
        
        # Prepare match confirmed data for both users
        inviter_data = {
            "room_id": room_id,
            "invitee_name": invitee_user.name,
            "session_id": inviter_session.id
        }
        
        invitee_data = {
            "room_id": room_id,
            "session_id": invitee_session.id
        }
        
        # Send confirmation with session IDs - use multiple methods to ensure delivery
        # Use 'to=' for direct socket targeting, user room, and session room as fallbacks
        inviter_sent = False
        inviter_user_room = f"user_{inviter_user_id_int}"
        
        # Verify inviter is in their user room before sending
        from flask_socketio import rooms as get_rooms
        try:
            if final_inviter_sid:
                inviter_rooms_before = get_rooms(final_inviter_sid)
                print(f"✓ Inviter is in rooms before send: {inviter_rooms_before}")
                if inviter_user_room not in inviter_rooms_before:
                    print(f"⚠ WARNING: Inviter not in user room {inviter_user_room}!")
                    print(f"⚠ Attempting to join inviter to room {inviter_user_room}...")
                    try:
                        join_room(inviter_user_room, sid=final_inviter_sid)
                        print(f"✓ Joined inviter to user room {inviter_user_room}")
                    except Exception as join_err:
                        print(f"✗ Could not join inviter to user room: {join_err}")
        except Exception as e:
            print(f"⚠ Could not verify inviter room membership: {e}")
        
        try:
            # Method 1: Send directly to socket ID
            if final_inviter_sid:
                print(f"📤 Attempting to send match_confirmed to inviter (method 1: direct socket):")
                print(f"  - Socket ID: {final_inviter_sid}")
                print(f"  - Data: {inviter_data}")
                socketio.emit('match_confirmed', inviter_data, to=final_inviter_sid)
                print(f"✓ Match confirmed sent to inviter (sid: {final_inviter_sid})")
                inviter_sent = True
            else:
                print(f"⚠ No valid inviter socket ID found for direct send")
            
            # Method 2: Send to inviter's user room (always works if user is registered)
            print(f"📤 Sending match_confirmed to inviter (method 2: user room {inviter_user_room})")
            socketio.emit('match_confirmed', inviter_data, room=inviter_user_room)
            print(f"✓ Match confirmed sent to inviter user room {inviter_user_room}")
            inviter_sent = True
            
            # Method 3: Also send to session room (inviter should be in it now)
            print(f"📤 Sending match_confirmed to inviter (method 3: session room {room_id})")
            socketio.emit('match_confirmed', inviter_data, room=room_id)
            print(f"✓ Match confirmed sent to session room {room_id}")
            inviter_sent = True
            
        except Exception as e:
            print(f"✗ Error sending match_confirmed to inviter: {e}")
            import traceback
            traceback.print_exc()
            # Still try user room as fallback
            try:
                socketio.emit('match_confirmed', inviter_data, room=inviter_user_room)
                print(f"✓ Fallback: Match confirmed sent to inviter user room after error")
                inviter_sent = True
            except Exception as e2:
                print(f"✗ Fallback also failed: {e2}")
        
        invitee_sent = False
        try:
            # Send directly to socket
            print(f"📤 Attempting to send match_confirmed to invitee:")
            print(f"  - Socket ID: {request.sid}")
            print(f"  - Data: {invitee_data}")
            socketio.emit('match_confirmed', invitee_data, to=request.sid)
            print(f"✓ Match confirmed sent to invitee (sid: {request.sid})")
            invitee_sent = True
            
            # Also send to user room as additional backup
            user_room = f"user_{invitee_user.id}"
            socketio.emit('match_confirmed', invitee_data, room=user_room)
            print(f"✓ Also sent to user room {user_room} as backup")
        except Exception as e:
            print(f"✗ Error sending match_confirmed to invitee: {e}")
            import traceback
            traceback.print_exc()
        
        # Method 4: ALWAYS also send to the session room as backup (both users are now in the room)
        # This ensures both users receive the event even if direct targeting fails
        print("📢 Broadcasting match_confirmed to session room as additional backup...")
        try:
            # Verify both users are in the room before sending
            from flask_socketio import rooms as get_rooms
            try:
                inviter_rooms_after = get_rooms(final_inviter_sid if final_inviter_sid else inviter_sid)
                invitee_rooms_after = get_rooms(request.sid)
                print(f"✓ Room membership before broadcast - Inviter rooms: {inviter_rooms_after}, Invitee rooms: {invitee_rooms_after}")
                if room_id not in inviter_rooms_after:
                    print(f"⚠ WARNING: Inviter not in session room {room_id} after join attempt!")
                    print(f"⚠ Attempting to re-join inviter to session room...")
                    try:
                        join_room(room_id, sid=final_inviter_sid)
                        print(f"✓ Re-joined inviter to session room {room_id}")
                    except Exception as rejoin_err:
                        print(f"✗ Could not re-join inviter: {rejoin_err}")
                if room_id not in invitee_rooms_after:
                    print(f"⚠ WARNING: Invitee not in session room {room_id} after join attempt!")
            except Exception as e:
                print(f"⚠ Could not verify room membership: {e}")
            
            # Send inviter data to room (inviter should receive this)
            socketio.emit('match_confirmed', inviter_data, room=room_id)
            print(f"✓ Inviter data broadcasted to session room {room_id}")
            
            # Send invitee data to room (invitee should receive this)
            socketio.emit('match_confirmed', invitee_data, room=room_id)
            print(f"✓ Invitee data broadcasted to session room {room_id}")
            
        except Exception as room_error:
            print(f"✗ Error broadcasting to session room: {room_error}")
            import traceback
            traceback.print_exc()
        
        print(f"✓ Match confirmed events sent (inviter: {inviter_sent}, invitee: {invitee_sent})")
        print(f"✓ Room ID: {room_id}")
        print(f"✓ Inviter session ID: {inviter_session.id if inviter_session else 'None'}")
        print(f"✓ Invitee session ID: {invitee_session.id if invitee_session else 'None'}")
        
    except Exception as e:
        print(f"✗ Error creating study sessions: {e}")
        import traceback
        traceback.print_exc()
        # Still send confirmation even if session creation fails
        try:
            socketio.emit('match_confirmed', {
                "room_id": room_id, 
                "invitee_name": invitee_user.name
            }, to=inviter_sid)
            print(f"✓ Match confirmed sent to inviter (fallback)")
        except Exception as emit_err:
            print(f"✗ Error sending fallback match_confirmed to inviter: {emit_err}")
        
        try:
            socketio.emit('match_confirmed', {
                "room_id": room_id
            }, to=request.sid)
            print(f"✓ Match confirmed sent to invitee (fallback)")
        except Exception as emit_err:
            print(f"✗ Error sending fallback match_confirmed to invitee: {emit_err}")

@socketio.on('disconnect')
def handle_disconnect():
    # Find user_id associated with the disconnected sid and remove them
    user_id_to_remove = None
    for user_id, sid in active_users.items():
        if sid == request.sid:
            user_id_to_remove = user_id
            break
    
    if user_id_to_remove:
        try:
            # Get the user from database
            user = User.query.get(int(user_id_to_remove))
            if user:
                # Set status to idle
                user.status = 'idle'
                user.study_goal = None
                user.goal_embedding = None
                
                # End any active study sessions and notify room members
                active_sessions = StudySession.query.filter_by(
                    user_id=user.id,
                    status='active'
                ).all()
                
                for session in active_sessions:
                    # Only mark as completed if session is older than 30 seconds
                    # This prevents brief disconnects from ending sessions
                    session_age = (datetime.now() - session.start_time).total_seconds()
                    if session_age < 30:
                        print(f"⚠ Session {session.id} is very new ({session_age:.1f}s), not marking as completed (might be brief disconnect)")
                        continue
                    
                    # Notify others in the room that user left
                    if session.room_id:
                        socketio.emit('receive_message', {
                            'id': f'system_{datetime.now().timestamp()}',
                            'sender': 'System',
                            'text': f'{user.name} has left the session.'
                        }, room=session.room_id)
                    
                    end_time = datetime.now()
                    duration = int((end_time - session.start_time).total_seconds())
                    session.end_time = end_time
                    session.duration_seconds = duration
                    session.status = 'completed'
                    print(f"✓ Ended session {session.id} for disconnected user {user_id_to_remove}")
                
                db.session.commit()
                print(f"✓ User {user_id_to_remove} status set to 'idle' and sessions ended")
            
            # Remove from active users
            del active_users[user_id_to_remove]
            print(f"User {user_id_to_remove} disconnected and was removed from active users.")
            print("Active users:", active_users)
        except Exception as e:
            print(f"✗ Error handling disconnect for user {user_id_to_remove}: {e}")
            import traceback
            traceback.print_exc()
            # Still remove from active_users even if DB update fails
            if user_id_to_remove in active_users:
                del active_users[user_id_to_remove]

@socketio.on('join_room')
def handle_join_room(data):
    room_id = data.get('room_id')
    if not room_id:
        print("✗ No room_id provided in join_room")
        return
        
    user_id = None
    for uid, sid in active_users.items():
        if sid == request.sid:
            try:
                user_id = int(uid)
            except:
                user_id = uid
            break
            
    if user_id:
        user = User.query.get(user_id)
        if user:
            try:
                join_room(room_id, sid=request.sid)
                print(f"✓ User {user.name} (id: {user_id}) joined room {room_id} via socket {request.sid}")
                
                # Verify they're in the room
                from flask_socketio import rooms as get_rooms
                user_rooms = get_rooms(request.sid)
                print(f"✓ User {user.name} is now in rooms: {user_rooms}")
                
                # Emit a 'system' message to the room
                socketio.emit('receive_message', {
                    'id': f'system_{datetime.now().timestamp()}',
                    'sender': 'System',
                    'text': f'{user.name} has joined the room.'
                }, room=room_id)
                print(f"✓ System message sent to room {room_id}")
            except Exception as e:
                print(f"✗ Error joining room: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ User {user_id} not found in database")
    else:
        print(f"✗ Could not find user_id for socket {request.sid} in join_room")

@socketio.on('leave_room')
def handle_leave_room(data):
    room_id = data.get('room_id')
    user_id = None
    for uid, sid in active_users.items():
        if sid == request.sid:
            user_id = uid
            break
    
    if user_id and room_id:
        user = User.query.get(user_id)
        if user:
            # Notify others in the room before leaving
            socketio.emit('receive_message', {
                'id': f'system_{datetime.now().timestamp()}',
                'sender': 'System',
                'text': f'{user.name} has left the session.'
            }, room=room_id)
            
            # Leave the room
            leave_room(room_id)
            print(f"User {user.name} left room {room_id}")
            
            # End the session
            session = StudySession.query.filter_by(
                user_id=user.id,
                room_id=room_id,
                status='active'
            ).first()
            
            if session:
                end_time = datetime.now()
                duration = int((end_time - session.start_time).total_seconds())
                session.end_time = end_time
                session.duration_seconds = duration
                session.status = 'completed'
                db.session.commit()
                print(f"✓ Ended session {session.id} for user {user.id}")

@socketio.on('request_google_meet')
def handle_request_google_meet(data):
    """Handle Google Meet request"""
    room_id = data.get('room_id')
    
    if not room_id:
        return
    
    # Find the requester
    requester_user_id = None
    requester_user = None
    for user_id, sid in active_users.items():
        if sid == request.sid:
            try:
                requester_user_id = int(user_id)
            except:
                requester_user_id = user_id
            requester_user = User.query.get(requester_user_id)
            break
    
    if not requester_user:
        return
    
    print(f"📹 Google Meet requested by {requester_user.name} (id: {requester_user_id}) in room {room_id}")
    
    # Initialize acceptances tracking for this room
    if room_id not in google_meet_acceptances:
        google_meet_acceptances[room_id] = {}
    
    # Reset acceptances for new request
    google_meet_acceptances[room_id] = {
        str(requester_user_id): True  # Requester auto-accepts
    }
    
    # Get all users in the room to send request only to others (not the requester)
    from flask_socketio import rooms as get_rooms
    request_data = {
        'requester_id': requester_user_id,
        'requester_name': requester_user.name
    }
    
    # Send request only to other users in the room (not the requester)
    for user_id_str, sid in active_users.items():
        try:
            user_rooms = get_rooms(sid)
            if room_id in user_rooms:
                try:
                    other_user_id = int(user_id_str)
                except:
                    other_user_id = user_id_str
                # Only send to other users, not the requester
                if other_user_id != requester_user_id:
                    socketio.emit('google_meet_requested', request_data, to=sid)
                    print(f"✓ Google Meet request sent to user {other_user_id} (socket {sid})")
        except:
            pass
    
    print(f"✓ Google Meet request sent to other users in room {room_id}")
    print(f"✓ Acceptances state: {google_meet_acceptances[room_id]}")

@socketio.on('accept_google_meet')
def handle_accept_google_meet(data):
    """Handle Google Meet acceptance"""
    room_id = data.get('room_id')
    
    if not room_id:
        return
    
    # Find the accepter
    accepter_user_id = None
    accepter_user = None
    for user_id, sid in active_users.items():
        if sid == request.sid:
            try:
                accepter_user_id = int(user_id)
            except:
                accepter_user_id = user_id
            accepter_user = User.query.get(accepter_user_id)
            break
    
    if not accepter_user:
        return
    
    print(f"✓ Google Meet accepted by {accepter_user.name} (id: {accepter_user_id}) in room {room_id}")
    
    # Initialize if not exists
    if room_id not in google_meet_acceptances:
        google_meet_acceptances[room_id] = {}
    
    # Mark this user as accepted
    google_meet_acceptances[room_id][str(accepter_user_id)] = True
    
    # Get all users in the room to check if both have accepted
    from flask_socketio import rooms as get_rooms
    room_sids = []
    for user_id_str, sid in active_users.items():
        try:
            user_rooms = get_rooms(sid)
            if room_id in user_rooms:
                room_sids.append((user_id_str, sid))
        except:
            pass
    
    print(f"✓ Users in room {room_id}: {[uid for uid, _ in room_sids]}")
    print(f"✓ Current acceptances: {google_meet_acceptances[room_id]}")
    
    # Check if all users in room have accepted
    all_accepted = all(
        google_meet_acceptances[room_id].get(str(uid), False) 
        for uid, _ in room_sids
    ) and len(google_meet_acceptances[room_id]) == len(room_sids)
    
    if all_accepted and len(room_sids) >= 2:
        # Both users have accepted - notify them to go to Google Meet
        print(f"✓ Both users accepted Google Meet request in room {room_id}")
        
        # Send notification to all users in the room that both have accepted
        socketio.emit('google_meet_accepted', {
            'all_accepted': True,
            'room_id': room_id
        }, room=room_id)
        print(f"✓ Google Meet acceptance confirmed - users can now go to meet.google.com")
        
        # Clear acceptances for this room
        if room_id in google_meet_acceptances:
            del google_meet_acceptances[room_id]
    else:
        # Notify that someone accepted (but not all yet)
        socketio.emit('google_meet_accepted', {
            'accepter_id': accepter_user_id,
            'accepter_name': accepter_user.name,
            'accepted_by_me': False
        }, room=room_id)
        print(f"✓ Waiting for other user(s) to accept... ({len(google_meet_acceptances[room_id])}/{len(room_sids)} accepted)")

@socketio.on('send_message')
def handle_send_message(data):
    room_id = data.get('room_id')
    text = data.get('text')
    
    print(f"📨 Message received - Room: {room_id}, Text: {text[:50]}...")
    
    user_id = None
    for uid, sid in active_users.items():
        if sid == request.sid:
            user_id = uid
            break
            
    if user_id:
        try:
            user_id_int = int(user_id)
        except:
            user_id_int = user_id
            
        user = User.query.get(user_id_int)
        if user and room_id and text:
            message_data = {
                'id': f'{user_id_int}_{datetime.now().timestamp()}',
                'sender': user.name,
                'text': text
            }
            
            # Broadcast the message to everyone in the room using 'room=' not 'to='
            print(f"📤 Broadcasting message to room {room_id} from {user.name} (id: {user_id_int})")
            print(f"  - Message data: {message_data}")
            
            # Send to the room
            socketio.emit('receive_message', message_data, room=room_id)
            print(f"✓ Message broadcasted to room {room_id}")
            
            # Also verify the sender is in the room
            from flask_socketio import rooms as get_rooms
            try:
                sender_rooms = get_rooms(request.sid)
                print(f"✓ Sender is in rooms: {sender_rooms}")
                if room_id not in sender_rooms:
                    print(f"⚠ WARNING: Sender not in room {room_id}! Joining now...")
                    join_room(room_id, sid=request.sid)
            except Exception as e:
                print(f"✗ Error checking sender rooms: {e}")
        else:
            print(f"✗ Invalid message data - user: {user}, room_id: {room_id}, text: {bool(text)}")
    else:
        print(f"✗ Could not find user_id for socket {request.sid}")

# --- DATABASE MIGRATION HELPER ---
def migrate_database():
    """Add new columns to User and StudySession tables if they don't exist"""
    import sqlite3
    
    db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
    if not os.path.exists(db_path):
        return  # Database will be created by create_all()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Migrate User table
    cursor.execute("PRAGMA table_info(user)")
    existing_user_columns = [column[1] for column in cursor.fetchall()]
    
    user_migrations = [
        ('preferred_session_length', "ALTER TABLE user ADD COLUMN preferred_session_length VARCHAR(20) DEFAULT 'medium'"),
        ('study_style', "ALTER TABLE user ADD COLUMN study_style VARCHAR(20) DEFAULT 'flexible'"),
        ('timezone', "ALTER TABLE user ADD COLUMN timezone VARCHAR(50)"),
        ('minimum_similarity_threshold', "ALTER TABLE user ADD COLUMN minimum_similarity_threshold REAL DEFAULT 0.3"),
        ('last_match_time', "ALTER TABLE user ADD COLUMN last_match_time DATETIME"),
        ('university', "ALTER TABLE user ADD COLUMN university VARCHAR(200)"),
        ('location', "ALTER TABLE user ADD COLUMN location VARCHAR(200)"),
        ('total_points', "ALTER TABLE user ADD COLUMN total_points INTEGER DEFAULT 0"),
        ('current_streak', "ALTER TABLE user ADD COLUMN current_streak INTEGER DEFAULT 0"),
        ('longest_streak', "ALTER TABLE user ADD COLUMN longest_streak INTEGER DEFAULT 0"),
        ('last_study_date', "ALTER TABLE user ADD COLUMN last_study_date DATE"),
        ('achievements', "ALTER TABLE user ADD COLUMN achievements TEXT")
    ]
    
    for column_name, sql in user_migrations:
        if column_name not in existing_user_columns:
            try:
                cursor.execute(sql)
                conn.commit()
                print(f"✓ Migrated: Added column '{column_name}' to user table")
            except sqlite3.OperationalError as e:
                print(f"✗ Migration error for '{column_name}': {e}")
    
    # Migrate StudySession table
    try:
        cursor.execute("PRAGMA table_info(study_session)")
        existing_session_columns = [column[1] for column in cursor.fetchall()]
        
        session_migrations = [
            ('session_rating', "ALTER TABLE study_session ADD COLUMN session_rating INTEGER"),
            ('matchmaking_rating', "ALTER TABLE study_session ADD COLUMN matchmaking_rating INTEGER"),
            ('feedback_text', "ALTER TABLE study_session ADD COLUMN feedback_text TEXT")
        ]
        
        for column_name, sql in session_migrations:
            if column_name not in existing_session_columns:
                try:
                    cursor.execute(sql)
                    conn.commit()
                    print(f"✓ Migrated: Added column '{column_name}' to study_session table")
                except sqlite3.OperationalError as e:
                    print(f"✗ Migration error for '{column_name}': {e}")
    except sqlite3.OperationalError:
        # Table might not exist yet, will be created by create_all()
        pass
    
    # Create friends table if it doesn't exist
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='friends'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE friends (
                    user_id INTEGER NOT NULL,
                    friend_id INTEGER NOT NULL,
                    PRIMARY KEY (user_id, friend_id),
                    FOREIGN KEY (user_id) REFERENCES user (id),
                    FOREIGN KEY (friend_id) REFERENCES user (id)
                )
            """)
            conn.commit()
            print(f"✓ Migrated: Created friends table")
    except sqlite3.OperationalError as e:
        print(f"✗ Migration error for friends table: {e}")
    
    # Create friend_request table if it doesn't exist
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='friend_request'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE friend_request (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    sender_id INTEGER NOT NULL,
                    receiver_id INTEGER NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at DATETIME NOT NULL,
                    FOREIGN KEY (sender_id) REFERENCES user (id),
                    FOREIGN KEY (receiver_id) REFERENCES user (id)
                )
            """)
            conn.commit()
            print(f"✓ Migrated: Created friend_request table")
    except sqlite3.OperationalError as e:
        print(f"✗ Migration error for friend_request table: {e}")
    
    conn.close()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    with app.app_context():
        # Create all tables (handles both SQLite and PostgreSQL)
        # This will create all tables with all columns defined in the models
        db.create_all()
        print(f"✓ Database initialized: {app.config['SQLALCHEMY_DATABASE_URI'][:50]}...")
    # Use socketio.run() instead of app.run()
    print("Starting Flask-SocketIO server with gevent...")
    socketio.run(app, debug=True, port=5000, log_output=True)