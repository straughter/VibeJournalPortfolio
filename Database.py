import os
import faiss
import pickle
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from firebase_admin import firestore, credentials, initialize_app, get_app
import logging
import datetime
from datetime import timezone, timedelta
from dotenv import load_dotenv

from utils.text_processing import normalize_keywords_spacy  # Updated import

from .article_processing import (
    filter_claude_response,
    fetch_articles_by_keywords_with_scoring
)  # Updated import from article_processing.py

from .synonyms import get_synonyms
import inflect
import spacy
import numpy as np
from urllib.parse import urlparse

# Import data models from models.py
from .models import (
    get_article_by_id,
    encode_text_to_vector_transformer,
    fetch_relevant_articles_with_faiss,
    # Your data models go here
    # For example:
    # ArticleModel,
    # UserModel,
    # etc.
)

# Import FAISS functions from faiss_utils.py
from .faiss_utils import (
    fetch_relevant_articles_with_faiss,
    fetch_relevant_summaries_with_faiss,
    add_article_to_faiss,
    add_summary_to_faiss,
    faiss_index_articles,
    faiss_index_summaries,
    article_id_mapping,
    summary_id_mapping,
    FAISS_INDEX_ARTICLES_PATH,
    ARTICLE_MAPPING_PATH,
    FAISS_INDEX_SUMMARIES_PATH,
    SUMMARY_MAPPING_PATH,
    model  # Importing the model instance if needed
)

# Continue with the rest of your code...

# ----------------------------
# 1. Setup and Initialization
# ----------------------------

# Initialize spaCy English model
nlp = spacy.load('en_core_web_sm')

# Initialize inflect engine for pluralization
p = inflect.engine()

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.debug(f"Loaded environment variables from {dotenv_path}")
else:
    logger.error(f".env file not found at path: {dotenv_path}")
    raise FileNotFoundError(f".env file not found at path: {dotenv_path}")

# Retrieve Firebase credentials path and other configs from environment variables
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH', 'utils/config/vibejournal-48877-firebase-adminsdk-by6i0-55488fc473.json')
FIREBASE_BUCKET = os.getenv('FIREBASE_BUCKET', 'your-default-bucket.appspot.com')
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', 'utils/faiss_index.bin')
SUMMARY_MAPPING_PATH = os.getenv('SUMMARY_MAPPING_PATH', 'utils/summary_mapping.pkl')  # Updated for summaries
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')

# Initialize Firebase Admin SDK if not already initialized
try:
    app = get_app()
    logger.debug("Firebase app already initialized.")
except ValueError:
    # Firebase app is not initialized, proceed to initialize
    credentials_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), FIREBASE_CREDENTIALS_PATH)
    if not os.path.exists(credentials_path):
        logger.error(f"Firebase credentials file not found at path: {credentials_path}")
        raise FileNotFoundError(f"Firebase credentials file not found at path: {credentials_path}")
    
    cred = credentials.Certificate(credentials_path)
    initialize_app(cred, {
        'storageBucket': FIREBASE_BUCKET
    })
    logger.debug("Firebase initialized successfully.")

# Initialize Firestore client
try:
    db = firestore.client()
    logger.debug("Firestore client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Firestore client: {e}")
    raise

# Initialize Sentence Transformer model
try:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.debug(f"SentenceTransformer model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    model = None  # Handle appropriately in your application

# Load FAISS index
if os.path.exists(FAISS_INDEX_PATH):
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logger.debug("FAISS index loaded successfully using faiss.read_index.")
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        faiss_index = None
else:
    logger.error(f"FAISS index file not found at path: {FAISS_INDEX_PATH}")
    faiss_index = None  # Handle this appropriately in your application

# Load summary ID to Firestore document mapping
if os.path.exists(SUMMARY_MAPPING_PATH):
    try:
        with open(SUMMARY_MAPPING_PATH, 'rb') as f:
            summary_id_mapping = pickle.load(f)  # Dict[int, str] mapping FAISS IDs to Firestore summary IDs
        logger.debug("Summary ID mapping loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading summary ID mapping: {e}")
        summary_id_mapping = {}
else:
    # Create the summary_mapping.pkl file if it doesn't exist
    try:
        os.makedirs(os.path.dirname(SUMMARY_MAPPING_PATH), exist_ok=True)
        with open(SUMMARY_MAPPING_PATH, 'wb') as f:
            pickle.dump({}, f)
        summary_id_mapping = {}
        logger.debug("Created new summary mapping file.")
    except Exception as e:
        logger.error(f"Error creating summary mapping file: {e}")
        summary_id_mapping = {}

# ----------------------------
# 2. Core Functions
# ---------------------------- 

def fetch_all_summaries() -> List[Dict[str, Any]]:
    """
    Fetches all summaries from Firestore.

    :return: A list of summary data dictionaries.
    """
    try:
        summaries_ref = db.collection('summaries').stream()
        summaries = []
        for doc in summaries_ref:
            summary_data = doc.to_dict()
            summary_data['id'] = doc.id  # Include the document ID
            summaries.append(summary_data)
        logger.debug(f"Fetched {len(summaries)} summaries from Firestore.")
        return summaries
    except Exception as e:
        logger.error(f"Error fetching all summaries: {e}")
        return []

def initialize_faiss_index():
    """
    Initializes FAISS index and creates necessary files if they don't exist.
    This function should be called after all necessary functions are defined.
    """
    global faiss_index, summary_id_mapping
    if faiss_index is None:
        try:
            dimension = 384  # Dimension for 'all-MiniLM-L6-v2' model
            faiss_index = faiss.IndexFlatL2(dimension)
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            logger.debug("Created new FAISS index.")
        except Exception as e:
            logger.error(f"Error creating new FAISS index: {e}")
            faiss_index = faiss.IndexFlatL2(384)  # Fallback
    
    # Load existing vectors if available
    try:
        summaries = fetch_all_summaries()
        vectors = [summary['vector'] for summary in summaries if 'vector' in summary]
        if vectors:
            vectors_np = np.array(vectors).astype('float32')
            faiss_index.add(vectors_np)
            faiss.write_index(faiss_index, FAISS_INDEX_PATH)
            logger.debug(f"FAISS index initialized with {faiss_index.ntotal} vectors.")
        else:
            logger.debug("No vectors found to add to FAISS index.")
    except Exception as e:
        logger.error(f"Error initializing FAISS index with vectors: {e}")

def sanitize_document_id(doc_id: str) -> str:
    """
    Sanitizes the document ID by replacing illegal characters.

    :param doc_id: The original document ID.
    :return: A sanitized document ID safe for use in Firestore.
    """
    # Replace forward slashes '/' and backslashes '\'
    return doc_id.replace('/', '_').replace('\\', '_')

def create_user(user_id: str, email: str, name: str) -> bool:
    """
    Creates a new user document in Firestore with default preferences and settings.
    
    :param user_id: The unique identifier for the user.
    :param email: The user's email address.
    :param name: The user's name.
    :return: True if the user is created successfully, False otherwise.
    """
    try:
        user_ref = db.collection('users').document(user_id)
        user_ref.set({
            'email': email,
            'name': name,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'therapeutic_goals': {
                'manage_anxiety': False,
                'increase_mindfulness': False,
                'improve_self_esteem': False,
                'reduce_stress': False,
                'preferred_outcomes': ''
            },
            'reminder_settings': {
                'reminder_time': 'Morning',
                'custom_reminder_time': '',
                'reminder_tone': 'gentle',
                'reflection_frequency': 'every_entry',
                'periodic_reflection_interval': 'weekly',
                'reflection_prompt_style': 'guided',
                'affirmation_frequency': 'with_every_entry',
                'periodic_affirmation_interval': 'weekly',
                'affirmation_style': 'direct'
            },
            'privacy_settings': {
                'anonymous_mode': False,
                'data_sharing_preferences': 'analytics'
            },
            'crisis_support_preferences': {
                'crisis_method': 'hotline'
            },
            'music_preferences': {
                'genres': [],
                'instrumental_vs_vocal': 'instrumental',
                'mood_specific_preferences': {}
            },
            'coping_mechanisms': {
                'coping_mechanisms': [],
                'emotional_triggers': []
            },
            'journaling_preferences': {
                'journaling_style': 'structured',
                'prompt_frequency': 'daily',
                'input_method': 'text',
                'follow_up_opt_in': False,
                'check_in_frequency': 3,
                'recent_follow_up_status': {
                    'last_prompt': '',
                    'last_session_continued': False
                }
            },
            'onboarding_preferences': {
                'coping_mechanisms': [],
                'emotional_triggers': []
            },
            # *** New: Audio Settings ***
            'audio_settings': {
                'voice': 'alloy',            # Default voice
                'language': 'en-US',         # Default language
                'autoPlayResponses': False   # Default playback setting
            },
            # *** New: User Feedback ***
            'user_feedback': []            # New field to store general user feedback
            # *** End of New: User Feedback ***
        })
        logger.debug(f"User {user_id} created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating user {user_id}: {e}")
        return False

def create_journal_entry(
    user_id: str,
    id: str,
    title: str,
    entry_text: str,
    font_style: str,
    emotions: List[Dict[str, Any]] = [],
    images: List[str] = [],
    theme: str = 'icebreaker_prompts',  # Added theme parameter with default value
    journal_data: Dict[str, Any] = {}
) -> bool:
    """
    Creates a new journal entry in Firestore under 'journal_history', 'emotional_history', and 'journal_entries' collections.
    """
    try:
        # Reference to the journal entry in 'journal_entries'
        journal_entries_ref = db.collection('users').document(user_id).collection('journal_entries').document(id)

        # Ensure journal_data contains all required fields
        journal_data.update({
            'title': title,
            'journal_text': entry_text if entry_text else 'No content provided.',
            'font_style': font_style,
            'emotions': [
                {
                    'emotion': emo['emotion'],
                    'percentage': emo['percentage']
                } for emo in emotions if 'emotion' in emo and 'percentage' in emo
            ],
            'images': images,
            'theme': theme,  # Include the theme here
            'is_deleted': False,
            'deleted_at': None,
            'createdAt': firestore.SERVER_TIMESTAMP,
            'updatedAt': firestore.SERVER_TIMESTAMP
        })

        # *** NEW: Filter the journal_text before saving ***
        if 'journal_text' in journal_data and isinstance(journal_data['journal_text'], str):
            original_text = journal_data['journal_text']
            filtered_text = filter_claude_response(original_text)
            journal_data['journal_text'] = filtered_text
            logger.debug(f"Filtered journal_text for entry {id}.")

        # Log journal entry creation details
        logger.debug(f"Creating journal entry with ID: {id}, Title: {title}, Theme: {theme}")

        # Write to 'journal_entries'
        journal_entries_ref.set(journal_data)
        logger.debug(f"Journal entry {id} for user {user_id} created successfully in 'journal_entries'.")

        # Write to 'journal_history'
        journal_ref = db.collection('users').document(user_id).collection('journal_history').document(id)
        journal_ref.set(journal_data)
        logger.debug(f"Journal entry {id} for user {user_id} created successfully in 'journal_history'.")

        # Prepare emotional history data
        emotional_data = {
            'emotions': journal_data['emotions'],
            'journal_entry_id': id,
            'theme': theme,  # Include the theme here
            'createdAt': firestore.SERVER_TIMESTAMP,
            'is_deleted': False,
            'deleted_at': None,
            'updatedAt': firestore.SERVER_TIMESTAMP
        }

        # Write to 'emotional_history'
        emotional_history_ref = db.collection('users').document(user_id).collection('emotional_history').document(id)
        emotional_history_ref.set(emotional_data)
        logger.debug(f"Emotional history entry {id} for user {user_id} created successfully.")

        return True
    except Exception as e:
        logger.error(f"Error creating journal entry {id} for user {user_id}: {e}")
        return False

def update_journal_context(user_id: str, entry_id: str, context: Dict[str, Any]) -> bool:
    """
    Updates the context of a specific journal entry for a user.

    :param user_id: The ID of the user.
    :param entry_id: The ID of the journal entry.
    :param context: The context data to update.
    :return: True if the update was successful, False otherwise.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_entries').document(entry_id)
        journal_ref.update({'context': context})
        return True
    except Exception as e:
        logger.error(f"Failed to update journal context for entry {entry_id}: {e}")
        return False

def set_mid_journal_check_in(user_id: str, entry_id: str, check_in_time: datetime) -> bool:
    """
    Sets a mid-journal check-in time for a specific journal entry.

    :param user_id: The ID of the user.
    :param entry_id: The ID of the journal entry.
    :param check_in_time: The datetime for the mid-journal check-in.
    :return: True if the operation was successful, False otherwise.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_entries').document(entry_id)
        journal_ref.update({'mid_journal_check_in': check_in_time})
        logger.info(f"Set mid_journal_check_in for entry {entry_id} of user {user_id} at {check_in_time}.")
        return True
    except Exception as e:
        logger.error(f"Failed to set mid_journal_check_in for entry {entry_id} of user {user_id}: {e}")
        return False 

def mark_entry_as_deleted(user_id: str, entry_id: str) -> bool:
    """
    Marks a journal entry as deleted (soft deletion) for a specific user.

    :param user_id: The ID of the user.
    :param entry_id: The ID of the journal entry.
    :return: True if the operation was successful, False otherwise.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_entries').document(entry_id)
        journal_ref.update({'is_deleted': True, 'deleted_at': datetime.datetime.utcnow()})
        logger.info(f"Marked entry {entry_id} as deleted for user {user_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to mark entry {entry_id} as deleted for user {user_id}: {e}")
        return False  

def recover_journal_entry(user_id: str, entry_id: str) -> bool:
    """
    Recovers a soft-deleted journal entry for a specific user.

    :param user_id: The ID of the user.
    :param entry_id: The ID of the journal entry.
    :return: True if the recovery was successful, False otherwise.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_entries').document(entry_id)
        journal_ref.update({'is_deleted': False, 'deleted_at': None})
        logger.info(f"Recovered journal entry {entry_id} for user {user_id}.")
        return True
    except Exception as e:
        logger.error(f"Failed to recover journal entry {entry_id} for user {user_id}: {e}")
        return False

def update_journal_entry(user_id: str, entry_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Updates specific fields of a journal entry for a user.

    :param user_id: The ID of the user.
    :param entry_id: The ID of the journal entry.
    :param update_data: A dictionary of fields to update with their new values.
    :return: True if the update was successful, False otherwise.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_entries').document(entry_id)
        journal_ref.update(update_data)
        logger.info(f"Updated journal entry {entry_id} for user {user_id} with data {update_data}.")
        return True
    except Exception as e:
        logger.error(f"Failed to update journal entry {entry_id} for user {user_id}: {e}")
        return False

def add_user_feedback(user_id: str, journal_id: str, feedback: str) -> bool:
    """
    Adds user feedback to both the journal entry and the user's general feedback list.

    :param user_id: The unique identifier for the user.
    :param journal_id: The unique identifier for the journal entry.
    :param feedback: The feedback content.
    :return: True if feedback is added successfully to both, False otherwise.
    """
    try:
        # Add feedback to journal entry
        journal_ref = db.collection('users').document(user_id).collection('journal_history').document(journal_id)
        journal_ref.update({
            'user_feedback_on_prompts': firestore.ArrayUnion([{
                'feedback': feedback,
                'feedback_type': 'journal',
                'createdAt': firestore.SERVER_TIMESTAMP
            }])
        })
        logger.debug(f"Added feedback to journal entry {journal_id} for user {user_id}.")

        # Add general feedback to user document
        user_ref = db.collection('users').document(user_id)
        user_ref.update({
            'user_feedback': firestore.ArrayUnion([{
                'feedback': feedback,
                'feedback_type': 'general',
                'createdAt': firestore.SERVER_TIMESTAMP
            }])
        })
        logger.debug(f"Added general feedback for user {user_id}.")

        return True
    except Exception as e:
        logger.error(f"Error adding feedback to journal entry {journal_id} and user {user_id}: {e}")
        return False

def add_reflection(user_id: str, id: str, reflection_id: str, content: str) -> bool:
    """
    Adds a reflection to a specific journal entry.

    :param user_id: The unique identifier for the user.
    :param id: The unique identifier for the journal entry.
    :param reflection_id: The unique identifier for the reflection.
    :param content: The content of the reflection.
    :return: True if the reflection is added successfully, False otherwise.
    """
    try:
        reflections_ref = db.collection('users').document(user_id).collection('journal_history').document(id).collection('reflections').document(reflection_id)

        reflection_data = {
            'content': content,
            'createdAt': firestore.SERVER_TIMESTAMP  # Server-side timestamp
        }

        # *** NEW: Filter the reflection content before saving ***
        if 'content' in reflection_data and isinstance(reflection_data['content'], str):
            original_content = reflection_data['content']
            filtered_content = filter_claude_response(original_content)
            reflection_data['content'] = filtered_content
            logger.debug(f"Filtered reflection content for reflection {reflection_id}.")

        reflections_ref.set(reflection_data)

        logger.debug(f"Reflection {reflection_id} added to journal entry {id}.")
        return True
    except Exception as e:
        logger.error(f"Error adding reflection {reflection_id} to journal entry {id}: {e}")
        return False

def update_user_preferences(user_id: str, preferences: Dict[str, Any]) -> bool:
    """
    Updates the user preferences in Firestore.

    :param user_id: The unique identifier for the user.
    :param preferences: A dictionary containing user preferences to update.
    :return: True if the update is successful, False otherwise.
    """
    try:
        user_ref = db.collection('users').document(user_id)
        user_ref.update(preferences)
        logger.debug(f"Updated preferences for user_id {user_id}: {preferences}")
        return True
    except Exception as e:
        logger.error(f"Error updating preferences for user_id {user_id}: {e}")
        return False

def create_playlist(user_id: str, playlist_id: str, playlist_name: str, mood: str, songs: List[Dict[str, Any]]) -> bool:
    """
    Creates a new playlist document in Firestore.

    :param user_id: The unique identifier for the user.
    :param playlist_id: The unique identifier for the playlist.
    :param playlist_name: The name of the playlist.
    :param mood: The mood associated with the playlist.
    :param songs: A list of songs to include in the playlist.
    :return: True if the playlist is created successfully, False otherwise.
    """
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        formatted_songs = [
            {
                'title': song.get('title', ''),
                'artist': song.get('artist', ''),
                'albumArt': song.get('albumArt', ''),
                'spotifyLink': song.get('spotifyLink', '')
            } for song in songs
        ]
        
        playlist_ref.set({
            'userId': user_id,
            'playlistName': playlist_name,
            'mood': mood,
            'songs': formatted_songs,
            'createdAt': firestore.SERVER_TIMESTAMP
        })
        logger.debug(f"Playlist {playlist_id} for user {user_id} created successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating playlist {playlist_id} for user {user_id}: {e}")
        return False

# ----------------------------
# 3. Summaries Collection Functions
# ---------------------------- 
def encode_text_to_vector_transformer(text: str) -> Optional[List[float]]:
    """
    Encodes text into a vector using the initialized SentenceTransformer model.

    Args:
        text (str): The text to encode.

    Returns:
        Optional[List[float]]: The embedding vector as a list of floats, or None if encoding fails.
    """
    if not model:
        logger.error("SentenceTransformer model is not initialized.")
        return None
    try:
        vector = model.encode(text, convert_to_numpy=True).astype('float32')
        logger.debug(f"Encoded text to vector with shape {vector.shape}.")
        return vector.tolist()
    except Exception as e:
        logger.error(f"Error encoding text to vector: {e}") 
        return None
def save_summary(user_id: str, summary_id: str, summary_text: str, sources: List[str]) -> bool:
    """
    Saves the summarized article to Firestore under the 'summaries' collection.

    Args:
        user_id (str): The ID of the user requesting the summary.
        summary_id (str): The unique ID of the summary.
        summary_text (str): The summarized text.
        sources (List[str]): List of source URLs.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # *** NEW: Filter the summary before saving ***
        filtered_summary = filter_claude_response(summary_text)
        logger.debug(f"Filtered summary for summary {summary_id}.")

        # Reference to the 'summaries' collection
        summary_ref = db.collection('summaries').document(summary_id)
        summary_ref.set({
            'user_id': user_id,
            'summary_text': filtered_summary,  # Use filtered summary
            'sources': sources,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        logger.debug(f"Summary {summary_id} saved successfully in 'summaries' collection.")

        # *** NEW: Encode the summary for FAISS ***
        summary_vector = encode_text_to_vector_transformer(filtered_summary)
        if summary_vector:
            add_summary_to_faiss(summary_id, summary_vector)
            logger.debug(f"Summary {summary_id} vector added to FAISS index.")
        else:
            logger.error(f"Failed to encode summary {summary_id} for FAISS indexing.")

        return True
    except Exception as e:
        logger.error(f"Error saving summary to Firestore: {e}")
        return False

def get_summary_by_id(summary_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a summary by its ID from the 'summaries' collection.

    Args:
        summary_id (str): The unique identifier for the summary.

    Returns:
        Optional[Dict[str, Any]]: Summary data if found, else None.
    """
    try:
        summary_ref = db.collection('summaries').document(summary_id)
        doc = summary_ref.get()
        if doc.exists:
            summary_data = doc.to_dict()
            summary_data['id'] = summary_id  # Include the document ID
            logger.debug(f"Summary {summary_id} retrieved successfully.")
            return summary_data
        else:
            logger.warning(f"Summary {summary_id} does not exist.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving summary {summary_id}: {e}")
        return None

def update_summary(summary_id: str, update_data: Dict[str, Any]) -> bool:
    """
    Updates specified fields of a summary.

    Args:
        summary_id (str): The unique identifier for the summary.
        update_data (Dict[str, Any]): Fields to update.

    Returns:
        bool: True if update is successful, False otherwise.
    """
    try:
        summary_ref = db.collection('summaries').document(summary_id)
        if not summary_ref.get().exists:
            logger.warning(f"Summary {summary_id} does not exist. Cannot update non-existent summary.")
            return False

        # *** NEW: Optionally Filter Updated Summary ***
        if 'summary_text' in update_data and isinstance(update_data['summary_text'], str):
            original_summary = update_data['summary_text']
            filtered_summary = filter_claude_response(original_summary)
            update_data['summary_text'] = filtered_summary
            logger.debug(f"Filtered updated summary for summary {summary_id}.")

        # *** NEW: Re-encode the summary if it has been updated ***
        if 'summary_text' in update_data:
            update_data['vector'] = encode_text_to_vector_transformer(update_data['summary_text'])
            logger.debug(f"Re-encoded vector for updated summary {summary_id}.")

        # Update the 'updated_at' timestamp
        update_data['updated_at'] = firestore.SERVER_TIMESTAMP

        summary_ref.update(update_data)
        logger.debug(f"Summary {summary_id} updated with data {update_data}")
        return True
    except Exception as e:
        logger.error(f"Error updating summary {summary_id}: {e}")
        return False

def delete_summary(summary_id: str) -> bool:
    """
    Deletes a summary by its ID.

    Args:
        summary_id (str): The unique identifier for the summary.

    Returns:
        bool: True if the deletion is successful, False otherwise.
    """
    try:
        summary_ref = db.collection('summaries').document(summary_id)
        # Check if the summary exists before deleting
        if not summary_ref.get().exists:
            logger.warning(f"Summary {summary_id} does not exist. Cannot delete non-existent summary.")
            return False
        summary_ref.delete()
        logger.debug(f"Summary {summary_id} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting summary {summary_id}: {e}")
        return False

def save_summary(user_id: str, 
                summary_id: str, 
                summary_text: str, 
                sources: List[str], 
                primary_mood: str = 'neutral',
                vector: Optional[List[float]] = None) -> bool:
    """
    Save the summarized article to Firestore under the 'summaries' collection.

    Args:
        user_id (str): The ID of the user.
        summary_id (str): The unique ID for the summary.
        summary_text (str): The summarized text.
        sources (List[str]): List of source URLs.
        primary_mood (str): Primary mood detected (default: 'neutral').
        vector (Optional[List[float]]): Encoded vector of the summary.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    try:
        # Access global FAISS index and mapping
        global faiss_index_summaries
        global summary_id_mapping

        # Filter the summary before saving
        filtered_summary = filter_claude_response(summary_text)
        logger.debug(f"Filtered summary for summary ID {summary_id}.")

        # Generate vector if not provided
        if vector is None and model is not None:
            vector = encode_text_to_vector_transformer(filtered_summary)
            logger.debug("Generated vector for summary text.")

        # Create 'summaries' collection if it doesn't exist (implicitly done by adding document)
        summary_ref = db.collection('summaries').document(summary_id)
        
        # Prepare summary data
        summary_data = {
            'user_id': user_id,
            'summary_text': filtered_summary,
            'sources': sources,
            'primary_mood': primary_mood,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }

        # Add vector if available
        if vector is not None:
            summary_data['vector'] = vector
            # Add to FAISS index with all required parameters
            if add_summary_to_faiss(
                summary_id=summary_id,
                vector=np.array(vector).astype('float32'),
                faiss_index=faiss_index_summaries,
                id_mapping=summary_id_mapping,
                mapping_path=SUMMARY_MAPPING_PATH,
                faiss_path=FAISS_INDEX_SUMMARIES_PATH
            ):
                logger.debug(f"Added summary vector to FAISS index for {summary_id}")
            else:
                logger.warning(f"Failed to add summary vector to FAISS index for {summary_id}")

        # Save to Firestore
        summary_ref.set(summary_data)
        logger.debug(f"Summary {summary_id} saved successfully in 'summaries' collection.")
        return True

    except Exception as e:
        logger.error(f"Error saving summary to Firestore: {e}")
        return False


# 4. Summaries and FAISS Integration

def encode_text_to_vector(text: str) -> Optional[List[float]]:
    """
    Encodes text into a vector using the SentenceTransformer model.

    Args:
        text (str): The text to encode.

    Returns:
        Optional[List[float]]: The embedding vector as a list of floats, or None if encoding fails.
    """
    if not model:
        logger.error("SentenceTransformer model is not initialized.")
        return None
    try:
        vector = model.encode(text, convert_to_numpy=True).astype('float32')
        logger.debug(f"Encoded text to vector with shape {vector.shape}.")
        return vector.tolist()
    except Exception as e:
        logger.error(f"Error encoding text to vector: {e}")
        return None

def fetch_relevant_summaries(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Fetches the top_k most relevant summaries based on the query using FAISS.

    Args:
        query (str): The search query.
        top_k (int): Number of top relevant summaries to fetch.

    Returns:
        List[Dict[str, Any]]: List of summary data dictionaries.
    """
    return fetch_relevant_summaries_with_faiss(query, top_k)


# 5. Emotion Aggregation Functions

def aggregate_emotions(user_id: str, period: str = 'week') -> Dict[str, float]:
    """
    Aggregates emotions over a specified period ('week' or 'month').

    :param user_id: The ID of the user.
    :param period: The aggregation period ('week' or 'month').
    :return: A dictionary with core emotions and their average percentages.
    """
    try:
        from collections import defaultdict

        if period == 'week':
            days = 7
        elif period == 'month':
            days = 30
        else:
            days = 7  # Default to week if invalid period is provided

        # Calculate the cutoff date
        cutoff_date = datetime.datetime.now(timezone.utc) - timedelta(days=days)

        # Query journal entries within the specified period that are not deleted
        journal_entries = db.collection('users').document(user_id).collection('journal_history') \
            .where('createdAt', '>=', cutoff_date) \
            .where('is_deleted', '==', False) \
            .stream()

        emotion_totals = defaultdict(float)
        entry_count = 0

        for entry in journal_entries:
            data = entry.to_dict()
            emotions = data.get('emotions', [])
            if not emotions:
                continue
            entry_count += 1
            for emotion in emotions:
                core_emotion = EmotionMapping.map_to_core(emotion['emotion'])
                if core_emotion:
                    emotion_totals[core_emotion] += emotion['percentage']

        # Calculate average percentages
        aggregated_emotions = {}
        for emotion, total in emotion_totals.items():
            aggregated_emotions[emotion] = round(total / entry_count, 2) if entry_count > 0 else 0

        logger.debug(f"Aggregated emotions for user {user_id} over {period}: {aggregated_emotions}")
        return aggregated_emotions

    except Exception as e:
        logger.error(f"Failed to aggregate emotions for user {user_id}: {e}")
        return {}

def store_aggregated_emotions(user_id: str, period: str, aggregated_data: Dict[str, float]) -> bool:
    """
    Stores the aggregated emotional data in Firestore.

    :param user_id: The ID of the user.
    :param period: The aggregation period ('week' or 'month').
    :param aggregated_data: The aggregated emotional data.
    :return: True if stored successfully, False otherwise.
    """
    try:
        aggregation_ref = db.collection('users').document(user_id).collection('aggregated_emotions').document(period)
        aggregation_ref.set({
            'emotions': aggregated_data,
            'generatedAt': firestore.SERVER_TIMESTAMP
        })
        logger.debug(f"Stored aggregated emotions for user {user_id} for period {period}.")
        return True
    except Exception as e:
        logger.error(f"Error storing aggregated emotions for user {user_id} for period {period}: {e}")
        return False

def aggregate_and_store_emotions(user_id: str, period: str = 'week') -> bool:
    """
    Aggregates emotions over a specified period and stores the results.

    :param user_id: The ID of the user.
    :param period: The aggregation period ('week' or 'month').
    :return: True if successful, False otherwise.
    """
    try:
        aggregated_emotions = aggregate_emotions(user_id, period)
        store_success = store_aggregated_emotions(user_id, period, aggregated_emotions)
        return store_success
    except Exception as e:
        logger.error(f"Failed to aggregate and store emotions for user {user_id} over period {period}: {e}")
        return False


# 6. Emotion Mapping Class


class EmotionMapping:
    """
    Maps derivative emotions to core emotions.
    """
    emotion_mapping = {
        # Happiness
        'joy': 'happiness',
        'contentment': 'happiness',
        'amusement': 'happiness',
        'love': 'happiness',
        'pride': 'happiness',
        'relief': 'happiness',
        'hope': 'happiness',
        'happiness': 'happiness',
        'inspiration': 'happiness',
        # Sadness
        'disappointment': 'sadness',
        'grief': 'sadness',
        'loneliness': 'sadness',
        'depression': 'sadness',
        'sadness': 'sadness',
        'hopelessness': 'sadness',
        'disheartened': 'sadness',
        'self-doubt': 'sadness',
        # Fear
        'anxiety': 'fear',
        'nervousness': 'fear',
        'terror': 'fear',
        'worry': 'fear',
        'panic': 'fear',
        'fear': 'fear',
        'stress': 'fear',
        # Disgust
        'contempt': 'disgust',
        'revulsion': 'disgust',
        'disapproval': 'disgust',
        'disgust': 'disgust',
        # Anger
        'frustration': 'anger',
        'irritation': 'anger',
        'rage': 'anger',
        'envy': 'anger',
        'anger': 'anger',
        'betrayal': 'anger',
        'embarrassment': 'anger',
        # Surprise
        'shock': 'surprise',
        'amazement': 'surprise',
        'astonishment': 'surprise',
        'surprise': 'surprise',
        # Trust
        'acceptance': 'trust',
        'admiration': 'trust',
        'friendliness': 'trust',
        'trust': 'trust',
        # Anticipation
        'interest': 'anticipation',
        'vigilance': 'anticipation',
        'curiosity': 'anticipation',
        'anticipation': 'anticipation',
        'excitement': 'anticipation',
    }

    @classmethod
    def map_to_core(cls, derivative_emotion: str) -> Optional[str]:
        """
        Maps a derivative emotion to a core emotion.

        :param derivative_emotion: The derivative emotion to map.
        :return: The core emotion if mapping exists, else None.
        """
        return cls.emotion_mapping.get(derivative_emotion.lower())


# 7. Permanently Delete Old Entries


def permanently_delete_old_entries() -> bool:
    """
    Permanently deletes journal entries that have been soft-deleted for more than 15 days.
    
    :return: True if the operation is successful, False otherwise.
    """
    try:
        # Calculate the cutoff date (entries deleted more than 15 days ago)
        cutoff_date = datetime.datetime.now(timezone.utc) - timedelta(days=15)

        # Query all users
        users_ref = db.collection('users').stream()
        deleted_count = 0

        for user in users_ref:
            user_id = user.id
            journal_entries = db.collection('users').document(user_id).collection('journal_history') \
                .where('is_deleted', '==', True) \
                .where('deleted_at', '<=', cutoff_date) \
                .stream()
            batch = db.batch()
            user_deleted_count = 0
            for entry in journal_entries:
                batch.delete(entry.reference)
                deleted_count += 1
                user_deleted_count += 1
            if user_deleted_count > 0:
                batch.commit()
                logger.debug(f"Permanently deleted {user_deleted_count} old journal entries for user {user_id}.")

        if deleted_count == 0:
            logger.debug("No old journal entries to delete.")
        
        return True
    except Exception as e:
        logger.error(f"Error permanently deleting old journal entries: {e}")
        return False

def permanently_delete_old_entries_task() -> bool:
    """
    Task to permanently delete old journal entries that have been soft-deleted for over 15 days.
    
    :return: True if deletion is successful, False otherwise.
    """
    try:
        success = permanently_delete_old_entries()
        if success:
            logger.debug("Successfully deleted old journal entries.")
        else:
            logger.error("Failed to delete old journal entries.")
        return success
    except Exception as e:
        logger.error(f"Error in permanently_delete_old_entries_task: {e}")
        return False

# 8. FAISS-Based Function
 


# 9. Article Management Functions
 

def create_article(article_id: str, article_data: Dict[str, Any], vector: Optional[List[float]] = None) -> bool:
    """
    Creates a new article document in Firestore with an optional vector field for FAISS.

    :param article_id: The unique identifier for the article.
    :param article_data: Dictionary containing article data.
    :param vector: Optional list of floats representing the article's vector embedding.
    :return: True if the article is created successfully, False otherwise.
    """
    try:
        # Sanitize the document ID
        sanitized_id = sanitize_document_id(article_id)
        articles_ref = db.collection('articles').document(sanitized_id)
        
        # Check if the article already exists
        if articles_ref.get().exists:
            logger.warning(f"Article with ID {sanitized_id} already exists. Skipping duplicate.")
            return False
        if not article_data.get('synopsis'):
            logger.error(f"No synopsis provided for article {article_id}. Cannot create article without synopsis.")
            return False
        # Prepare article data with new fields
        def ensure_list(field_value):
            if isinstance(field_value, str):
                return [item.strip() for item in field_value.replace(';', ',').split(',') if item.strip()]
            elif isinstance(field_value, list):
                return field_value
            else:
                return []
        
        # Process fields with added checks
        authors = ensure_list(article_data.get('authors', []))
        keywords = ensure_list(article_data.get('keywords', []))
        main_findings = ensure_list(article_data.get('main_findings', []))
        treatment_types = ensure_list(article_data.get('treatment_types', []))
        target_conditions = ensure_list(article_data.get('target_conditions', []))
        moods = ensure_list(article_data.get('moods', []))
        contexts = ensure_list(article_data.get('contexts', []))
        suggested_actions = ensure_list(article_data.get('suggested_actions', []))
        
        # Lowercase fields with safe checks
        keywords_lowercase = [k.lower() for k in keywords if isinstance(k, str)]
        moods_lowercase = [m.lower() for m in moods if isinstance(m, str)]
        contexts_lowercase = [c.lower() for c in contexts if isinstance(c, str)]
        suggested_actions_lowercase = [a.lower() for a in suggested_actions if isinstance(a, str)]
        
        # Combine all search terms into a single list without duplicates
        base_search_terms = list(set(
            keywords_lowercase +
            moods_lowercase +
            contexts_lowercase +
            suggested_actions_lowercase
        ))
        
        # *** NEW: Integrate Synonyms into search_terms_lowercase ***
        synonyms_terms = []
        for term in base_search_terms:
            synonyms = get_synonyms(term)
            synonyms_lower = [syn.lower() for syn in synonyms if isinstance(syn, str)]
            synonyms_terms.extend(synonyms_lower)
        
        # Combine base terms with synonyms, remove duplicates
        search_terms_lowercase = list(set(base_search_terms + synonyms_terms))
        
        # *** NEW: Limit the number of search terms to Firestore constraints ***
        MAX_SEARCH_TERMS = 100  # Adjust based on your needs
        if len(search_terms_lowercase) > MAX_SEARCH_TERMS:
            search_terms_lowercase = search_terms_lowercase[:MAX_SEARCH_TERMS]
            logger.warning(f"search_terms_lowercase exceeded {MAX_SEARCH_TERMS} terms. Truncated to fit Firestore constraints.")
        
        # Additional permutations for better matching
        additional_terms = []
        for term in base_search_terms:
            # Split phrases into individual words and add them as separate terms
            words = term.split()
            if len(words) > 1:
                additional_terms.extend([word.lower() for word in words])
        
        search_terms_lowercase.extend(additional_terms)
        search_terms_lowercase = list(set(search_terms_lowercase))  # Remove duplicates again
        
        # Lemmatize all search terms
        search_terms_lowercase = normalize_keywords_spacy(search_terms_lowercase)
        logger.debug(f"Lemmatized Search Terms: {search_terms_lowercase}")
        
        # Add plural forms
        expanded_search_terms = set(search_terms_lowercase)
        for term in search_terms_lowercase:
            plural = inflect.engine().plural(term)
            if plural != term:
                expanded_search_terms.add(plural)
        
        search_terms_lowercase = list(expanded_search_terms)
        logger.debug(f"Search Terms after Pluralization: {search_terms_lowercase}")
        
        # Retrieve 'pageUrl' and 'imageUrl' from article_data
        page_url = article_data.get('pageUrl', '')
        image_url = article_data.get('imageUrl', '')
        
        # *** NEW: Define a Default Image URL ***
        DEFAULT_IMAGE_URL = "https://example.com/default-image.jpg"  # Replace with your actual default image URL
        
        # Validate DOI with safe check
        doi = article_data.get('doi', None)
        if doi is not None and isinstance(doi, str):
            doi = doi.strip()  # Only call strip if doi is not None
            if doi.lower() == 'no doi':
                doi = ''
        else:
            doi = ''  # Set to empty if None or not a string
        
        # Validate 'pmcid'
        pmcid = article_data.get('pmcid', '')
        if pmcid and isinstance(pmcid, str):
            if pmcid.lower() in ['no pmcid', 'null']:
                pmcid = ''
            else:
                pmcid = pmcid.strip()
        else:
            pmcid = ''
        
        # Validate 'imageUrl' and set a default if invalid
        if image_url:
            def is_valid_url(url):
                try:
                    result = urlparse(url)
                    return all([result.scheme, result.netloc])
                except ValueError:
                    return False

            if not is_valid_url(image_url):
                logger.warning(f"Invalid imageUrl format for article {sanitized_id}: {image_url}. Setting 'imageUrl' to default.")
                image_url = DEFAULT_IMAGE_URL
        else:
            logger.info(f"No imageUrl provided for article {sanitized_id}. Setting to default.")
            image_url = DEFAULT_IMAGE_URL
        
        # *** NEW: Optionally Filter Other Fields (e.g., synopsis) ***
        if 'synopsis' in article_data and isinstance(article_data['synopsis'], str):
            original_synopsis = article_data['synopsis']
            filtered_synopsis = filter_claude_response(original_synopsis)
            article_data['synopsis'] = filtered_synopsis
            logger.debug(f"Filtered synopsis for article {sanitized_id}.")

        # Ensure mandatory fields are present
        treatment_types = treatment_types if treatment_types else []
        target_conditions = target_conditions if target_conditions else []
        
        # *** NEW: Define the article record with all required fields ***
        article_record = {
            'title': article_data.get('title', ''),
            'authors': authors,
            'publication_date': article_data.get('publication_date', ''),
            'doi': doi,
            'pmcid': pmcid,
            'sentiment_score': article_data.get('sentiment_score', 0.0),
            'synopsis': article_data.get('synopsis', ''),
            'keywords': keywords,
            'keywords_lowercase': keywords_lowercase,
            'main_findings': main_findings,
            'treatment_types': treatment_types,
            'target_conditions': target_conditions,
            'created_at': firestore.SERVER_TIMESTAMP,
            'moods': moods,
            'moods_lowercase': moods_lowercase,
            'contexts': contexts,
            'contexts_lowercase': contexts_lowercase,
            'suggested_actions': suggested_actions,
            'suggested_actions_lowercase': suggested_actions_lowercase,
            'search_terms_lowercase': search_terms_lowercase,
            'pageUrl': page_url,  # Added 'pageUrl' field
            'imageUrl': image_url  # Added 'imageUrl' field
        }
        
        # *** NEW: Include vector if provided ***
        if vector:
            article_record['vector'] = vector  # Add the vector field
        
        articles_ref.set(article_record)
        logger.debug(f"Article {sanitized_id} created successfully with vector: {bool(vector)}.")
        return True
    except Exception as e:
        logger.error(f"Error creating article {article_id}: {e}")
        return False
def update_article(article_id: str, update_data: Dict[str, Any], vector: Optional[List[float]] = None) -> bool:
    """
    Updates specified fields of an article, including the vector field if provided.

    :param article_id: The unique identifier for the article.
    :param update_data: Dictionary containing the fields to update.
    :param vector: Optional list of floats representing the updated vector embedding.
    :return: True if the update is successful, False otherwise.
    """
    try:
        sanitized_id = sanitize_document_id(article_id)
        article_ref = db.collection('articles').document(sanitized_id)
        # Check if the article exists before updating
        if not article_ref.get().exists:
            logger.warning(f"Article {sanitized_id} does not exist. Cannot update non-existent article.")
            return False
        
        # *** NEW: Include vector if provided ***
        if vector:
            update_data['vector'] = vector  # Add or update the vector field
        
        # *** Validation: If 'imageUrl' is being updated, ensure it's valid ***
        if 'imageUrl' in update_data:
            image_url = update_data['imageUrl'].strip()
            if image_url:
                def is_valid_url(url):
                    try:
                        result = urlparse(url)
                        return all([result.scheme, result.netloc])
                    except ValueError:
                        return False

                if not is_valid_url(image_url):
                    logger.warning(f"Invalid imageUrl format for article {sanitized_id}: {image_url}. Removing 'imageUrl' from update.")
                    del update_data['imageUrl']
        
        # *** NEW: Optionally Filter Updated Synopsis ***
        if 'synopsis' in update_data and isinstance(update_data['synopsis'], str):
            original_synopsis = update_data['synopsis']
            filtered_synopsis = filter_claude_response(original_synopsis)
            update_data['synopsis'] = filtered_synopsis
            logger.debug(f"Filtered updated synopsis for article {sanitized_id}.")

        # *** NEW: Re-encode the synopsis if it has been updated ***
        if 'synopsis' in update_data:
            update_data['vector'] = encode_text_to_vector_transformer(update_data['synopsis'])
            logger.debug(f"Re-encoded vector for updated article {sanitized_id}.")

        # *** NEW: Update the 'updated_at' timestamp ***
        update_data['updated_at'] = firestore.SERVER_TIMESTAMP

        article_ref.update(update_data)
        logger.debug(f"Article {sanitized_id} updated with data {update_data}")
        return True
    except Exception as e:
        logger.error(f"Error updating article {article_id}: {e}")
        return False

def get_article_by_id(article_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves an article by its ID.

    :param article_id: The unique identifier for the article.
    :return: A dictionary of the article data, or None if not found.
    """
    try:
        sanitized_id = sanitize_document_id(article_id)
        article_ref = db.collection('articles').document(sanitized_id)
        doc = article_ref.get()
        if doc.exists:
            article_data = doc.to_dict()
            article_data['id'] = sanitized_id  # Include the document ID
            logger.debug(f"Article {sanitized_id} retrieved successfully.")
            return article_data
        else:
            logger.warning(f"Article {sanitized_id} does not exist.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving article {article_id}: {e}")
        return None

def delete_article(article_id: str) -> bool:
    """
    Deletes an article by its ID.

    :param article_id: The unique identifier for the article.
    :return: True if the deletion is successful, False otherwise.
    """
    try:
        sanitized_id = sanitize_document_id(article_id)
        article_ref = db.collection('articles').document(sanitized_id)
        # Check if the article exists before deleting
        if not article_ref.get().exists:
            logger.warning(f"Article {sanitized_id} does not exist. Cannot delete non-existent article.")
            return False
        article_ref.delete()
        logger.debug(f"Article {sanitized_id} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting article {article_id}: {e}")
        return False

# *** ADDED: fetch_all_articles function ***
def fetch_all_articles() -> List[Dict[str, Any]]:
    """
    Fetches all articles from the Firestore 'articles' collection.

    Returns:
        List[Dict[str, Any]]: List of articles as dictionaries.
    """
    try:
        articles_ref = db.collection('articles')
        docs = articles_ref.stream()
        articles = []
        for doc in docs:
            article = doc.to_dict()
            article['id'] = doc.id  # Add the document ID to the article data
            articles.append(article)
            logger.debug(f"Fetched Article ID: {article['id']}, Synopsis: {article.get('synopsis', '')}")
        logger.debug(f"Fetched {len(articles)} articles from Firestore.")
        return articles
    except Exception as e:
        logger.error(f"Error fetching articles from Firestore: {e}")
        return []

# *** ADDED: fetch_all_summaries function ***
def fetch_all_summaries() -> List[Dict[str, Any]]:
    """
    Fetches all summaries from Firestore.

    :return: A list of summary data dictionaries.
    """
    try:
        summaries_ref = db.collection('summaries').stream()
        summaries = []
        for doc in summaries_ref:
            summary_data = doc.to_dict()
            summary_data['id'] = doc.id  # Include the document ID
            summaries.append(summary_data)
        logger.debug(f"Fetched {len(summaries)} summaries from Firestore.")
        return summaries
    except Exception as e:
        logger.error(f"Error fetching all summaries: {e}")
        return []

def update_analysis_result(article_id: str, analysis_data: Dict[str, Any]) -> bool:
    """
    Updates the analysis result for a specific article in Firestore.

    :param article_id: The unique identifier for the article.
    :param analysis_data: A dictionary containing analysis results to update.
    :return: True if the update is successful, False otherwise.
    """
    try:
        sanitized_id = sanitize_document_id(article_id)  # Corrected to use sanitize_document_id
        article_ref = db.collection('articles').document(sanitized_id)
        
        # Check if the article exists
        if not article_ref.get().exists:
            logger.warning(f"Article {sanitized_id} does not exist. Cannot update analysis result.")
            return False
        
        # Update the analysis_result field
        article_ref.update({'analysis_result': analysis_data})
        logger.debug(f"Analysis result for article {sanitized_id} updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error updating analysis result for article {article_id}: {e}")
        return False

def get_journal_entry_by_id(user_id: str, journal_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific journal entry by user ID and journal entry ID.

    :param user_id: The unique identifier for the user.
    :param journal_id: The unique identifier for the journal entry.
    :return: A dictionary containing the journal entry data, or None if not found.
    """
    try:
        journal_ref = db.collection('users').document(user_id).collection('journal_history').document(journal_id)
        doc = journal_ref.get()
        if doc.exists:
            journal_data = doc.to_dict()
            journal_data['id'] = journal_id  # Include the document ID
            logger.debug(f"Journal entry {journal_id} for user {user_id} retrieved successfully.")
            return journal_data
        else:
            logger.warning(f"Journal entry {journal_id} for user {user_id} does not exist.")
            return None
    except Exception as e:
        logger.error(f"Error retrieving journal entry {journal_id} for user {user_id}: {e}")
        return None



    # Add a reflection to the journal entry
    if add_reflection(user_id, journal_id, 'reflection_789', 'This is my first reflection'):
        logger.info("Reflection has been successfully added to Firestore.")
    else:
        logger.error("Failed to add reflection.")

    # Update user preferences
    if update_user_preferences(user_id, {
        'therapeutic_goals.manage_anxiety': True,
        'reminder_settings.reminder_time': 'Evening',
        # *** Updating Audio Settings ***
        'audio_settings.voice': 'echo',
        'audio_settings.language': 'es-ES',
        'audio_settings.autoPlayResponses': True
        # *** End of Updating Audio Settings ***
    }):
        logger.info("User preferences have been successfully updated in Firestore.")
    else:
        logger.error("Failed to update user preferences.")

    # Save a summary for an article
    article_id = 'article_789'
    summary_text = 'This is a summarized version of the article.'
    if save_summary(user_id, article_id, summary_text, sources=['https://example.com/article']):
        logger.info("Summary has been successfully saved to Firestore.")
    else:
        logger.error("Failed to save summary.")

    # *** NEW: Sample Usage for Articles Collection ***
    # Define the article data based on the provided summary
    new_article_id = '10.1016/j.psc.2017.08.008'  # Using DOI as ID
    article_data = {
        'title': "Mindfulness-Based Interventions for Anxiety and Depression",
        'authors': ["Stefan G. Hofmann", "Angelina F. Gómez"],
        'publication_date': "2017-09-18",
        'doi': "10.1016/j.psc.2017.08.008",
        'pmcid': "PMC5679245",
        'sentiment_score': 0.75,
        'synopsis': """
        Recent research has shown that mindfulness-based interventions (MBIs), such as Mindfulness-Based Stress Reduction (MBSR) and Mindfulness-Based Cognitive Therapy (MBCT), can effectively help people manage anxiety and depression. These techniques encourage individuals to focus on the present moment and respond to their thoughts and feelings without judgment.
        
        **Key Benefits:**
        - **Reduces Symptoms:** Studies indicate that MBIs significantly decrease anxiety and depression symptoms in a variety of people seeking treatment.
        - **Stronger than Non-Evidence-Based Treatments:** MBIs consistently outperform less effective therapies like relaxation training and supportive counseling.
        - **Compatibility with CBT:** The principles of MBIs align well with cognitive-behavioral therapy (CBT), making them a powerful option for improving mental health.
        
        The overarching idea is that by practicing mindfulness—through activities like meditation and yoga—individuals become less reactive to stressors and better equipped to handle their emotions. Overall, mindfulness can lead to lasting positive changes in mental well-being.
        """,
        'keywords': [
            "Mindfulness",
            "Mindfulness-Based Interventions (MBIs)",
            "Anxiety",
            "Depression",
            "Cognitive-Behavioral Therapy (CBT)",
            "Mindfulness-Based Stress Reduction (MBSR)",
            "Mindfulness-Based Cognitive Therapy (MBCT)"
        ],
        'main_findings': [
            "MBIs have demonstrated efficacy in alleviating anxiety and depression symptom severity.",
            "The principles of MBIs align with standard CBT practices, targeting emotional awareness and regulation.",
            "Mindfulness promotes a state of nonjudgmental awareness, which is beneficial for psychological well-being.",
            "MBSR and MBCT are the most common mindfulness-based treatments explored.",
            "Mindfulness practices can reduce symptoms of chronic pain, stress, and enhance quality of life."
        ],
        'treatment_types': [
            "Mindfulness-Based Stress Reduction (MBSR)",
            "Mindfulness-Based Cognitive Therapy (MBCT)"
        ],
        'target_conditions': [
            "Anxiety",
            "Depression"
        ],
        # *** NEW: Adding 'pageUrl' and 'imageUrl' ***
        'pageUrl': 'https://example.com/mindfulness-based-interventions-anxiety-depression',  # New field
        'imageUrl': "https://thoughtsonlifeandlove.com/wp-content/uploads/2024/10/istockphoto-1033774292-2048x2048-1-optimized.jpg",  # New field
        # *** End of NEW: Adding 'pageUrl' and 'imageUrl' ***
    }

    # *** NEW: Filter the article synopsis before generating vector ***
    if 'synopsis' in article_data and isinstance(article_data['synopsis'], str):
        original_synopsis = article_data['synopsis']
        filtered_synopsis = filter_claude_response(original_synopsis)
        article_data['synopsis'] = filtered_synopsis
        logger.debug(f"Filtered synopsis for article {new_article_id}.")
    # *** End of NEW: Filter the article synopsis before generating vector ***

    # Generate vector for the article synopsis
    article_vector = encode_text_to_vector_transformer(article_data['synopsis'])

    # *** UPDATED: Use the refactored create_article function ***
    if create_article(new_article_id, article_data, vector=article_vector):
        logger.info(f"Article '{article_data['title']}' has been successfully added to Firestore.")
    else:
        logger.error(f"Failed to add article '{article_data['title']}' to Firestore.")

    # Example: Retrieve the article
    retrieved_article = get_article_by_id(new_article_id)
    if retrieved_article:
        logger.info(f"Retrieved Article: {retrieved_article['title']}")
        # *** NEW: Example of accessing 'pageUrl' and 'imageUrl' ***
        page_url = retrieved_article.get('pageUrl', '')
        image_url = retrieved_article.get('imageUrl', '')
        if page_url:
            logger.info(f"Article Page URL: {page_url}")
        else:
            logger.info("No Page URL available for this article.")

        if image_url:
            logger.info(f"Article Image URL: {image_url}")
        else:
            logger.info("No Image URL available for this article.")
        # *** End of NEW: Example of accessing 'pageUrl' and 'imageUrl' ***
    else:
        logger.error(f"Article '{new_article_id}' could not be retrieved.")

    # Example: Update the sentiment score and imageUrl
    if retrieved_article:
        updated_vector = encode_text_to_vector_transformer(retrieved_article['synopsis'])
    else:
        updated_vector = []
    if update_article(new_article_id, {'sentiment_score': 0.80, 'imageUrl': "https://example.com/new-image.jpg"}, vector=updated_vector):
        logger.info(f"Article '{new_article_id}' sentiment score and imageUrl updated successfully.")
    else:
        logger.error(f"Failed to update sentiment score and imageUrl for article '{new_article_id}'.")

    # *** End of New: Sample Usage for Articles Collection ***

    # Permanently delete entries older than 15 days (this would typically be scheduled)
    if permanently_delete_old_entries():
        logger.info("Old journal entries permanently deleted successfully.")
    else:
        logger.error("Failed to permanently delete old journal entries.")
    
    # *** NEW: Validate and Update All Existing Articles ***
    # Uncomment the following line to run validation on all existing articles
    # validate_and_update_all_articles()
