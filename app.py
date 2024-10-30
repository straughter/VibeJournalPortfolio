import sys
import os
import json
import uuid  # For generating UUIDs
import logging
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler  # For scheduled tasks
from dotenv import load_dotenv 
from pydantic import ValidationError
from werkzeug.utils import secure_filename
from scrapy import signals
from scrapy_project.scrapy_project.spiders.mental_health_spider import MentalHealthSpider

from utils.firebase_init import db, bucket  
from utils.synonyms import get_synonyms 

# Ensure the project root directory is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging once in app.py
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Third-party imports (should come after standard and local imports)
import firebase_admin

# Local module imports
from services.diffbot_service import fetch_articles
# Removed the following import as we are using GPTSummarizer instead
# from services.summarization_service import summarize_text, save_summary

from utils.icebreaker_prompts import icebreaker_prompts_bp, generate_icebreaker_prompts
from utils.adhd_management_prompts import adhd_management_bp, generate_adhd_management_prompts
from utils.gratitude_journal_prompts import generate_gratitude_journal_prompts, gratitude_journal_bp
from utils.anxiety_management_prompts import anxiety_management_bp, generate_anxiety_management_prompts
from utils.self_compassion_prompts import generate_self_compassion_prompts, self_compassion_bp
from utils.motivation_goal_setting_prompts import generate_motivation_goal_setting_prompts, motivation_goal_setting_bp
from utils.sleep_quality_tracker_prompts import generate_sleep_quality_tracker_prompts, sleep_quality_tracker_bp
from utils.mindfulness_practice_prompts import generate_mindfulness_practice_prompts, mindfulness_practice_bp
from utils.relationship_building_prompts import generate_relationship_building_prompts, relationship_building_bp
from utils.personal_growth_prompts import generate_personal_growth_prompts, personal_growth_bp
from utils.creativity_inspiration_prompts import creativity_inspiration_bp, generate_creativity_inspiration_prompts
from EmotionalHistory import EmotionalHistory, JournalEntry, Emotion
from utils.emotion_detector import analyze_journal_entry 
from utils.filter import filter_claude_response
from services.spotify_playlist import simulate_spotify_playlist
from utils.user_preferences import get_user_preferences, update_user_preferences
from utils.notification_service import schedule_reminder 
from utils.article_filter import fetch_relevant_summaries, calculate_user_sentiment  # Updated import
from utils.database import (
    create_journal_entry,
    add_reflection,
    add_user_feedback,
    update_analysis_result,
    update_journal_context,
    set_mid_journal_check_in,
    mark_entry_as_deleted,
    update_journal_entry,
    recover_journal_entry,
    permanently_delete_old_entries,
    save_summary,  # Ensure this refers to the correct 'save_summary' if needed
    fetch_articles_by_keywords_with_scoring  # Added
)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}) 

# Register Blueprints 
app.register_blueprint(icebreaker_prompts_bp)
app.register_blueprint(adhd_management_bp)
app.register_blueprint(gratitude_journal_bp)
app.register_blueprint(anxiety_management_bp)
app.register_blueprint(self_compassion_bp)
app.register_blueprint(motivation_goal_setting_bp)
app.register_blueprint(sleep_quality_tracker_bp)
app.register_blueprint(mindfulness_practice_bp)
app.register_blueprint(relationship_building_bp)
app.register_blueprint(personal_growth_bp)
app.register_blueprint(creativity_inspiration_bp)

# Initialize GPTSummarizer
from utils.article_summarizer import GPTSummarizer  
summarizer = GPTSummarizer()

# Scheduler for automatic permanent deletion
scheduler = BackgroundScheduler()
scheduler.add_job(func=permanently_delete_old_entries, trigger="interval", hours=24)  # Runs every 24 hours
scheduler.start()

# Gracefully shut down the scheduler on application exit
import atexit
atexit.register(lambda: scheduler.shutdown())

def sanitize_article_id(article_id: str) -> str:
    """
    Sanitizes the article_id by replacing slashes with underscores.
    
    :param article_id: The original article ID (possibly containing slashes).
    :return: A sanitized article ID without slashes.
    """
    sanitized_id = article_id.replace('/', '_')
    logger.debug(f"Sanitized article_id from '{article_id}' to '{sanitized_id}'")
    return sanitized_id


#       New Endpoints            #


@app.route('/get_articles', methods=['GET'])
def get_articles():
    """
    Endpoint to fetch paginated summaries from Firestore.
    """
    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        if page < 1 or page_size < 1:
            raise ValueError("Page and page_size must be positive integers.")

        summaries_ref = db.collection('summaries').order_by('created_at', direction=db.Query.DESCENDING)
        
        try:
            total_summaries = summaries_ref.count().get().results[0].value
        except AttributeError:
            # Fallback if count() isn't supported
            total_summaries = len(list(summaries_ref.stream()))

        summaries = summaries_ref.offset((page - 1) * page_size).limit(page_size).stream()

        summaries_list = []
        for summary in summaries:
            summary_dict = summary.to_dict()
            summary_dict['id'] = summary.id
            summaries_list.append(summary_dict)

        return jsonify({
            'success': True,
            'data': {
                'total': total_summaries,
                'page': page,
                'page_size': page_size,
                'summaries': summaries_list  # Changed 'articles' to 'summaries'
            }
        }), 200

    except ValueError as ve:
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        logger.exception(f"Exception occurred while fetching summaries: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/get_summary/<summary_id>', methods=['GET'])
def get_summary(summary_id):
    """
    Endpoint to fetch a single summary by its ID.
    """
    try:
        sanitized_id = sanitize_article_id(summary_id)
        summary_ref = db.collection('summaries').document(sanitized_id)
        doc = summary_ref.get()
        if doc.exists:
            summary = doc.to_dict()
            summary['id'] = doc.id
            logger.info(f"Fetched summary '{summary.get('synopsis', 'No Synopsis')}' successfully.")
            return jsonify({'summary': summary}), 200
        else:
            logger.warning(f"Summary '{sanitized_id}' does not exist.")
            return jsonify({'error': 'Summary not found'}), 404
    except Exception as e:
        logger.exception(f"Exception occurred while fetching summary '{summary_id}': {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/search-summaries', methods=['POST'])
def search_summaries():
    """
    Endpoint to search summaries based on a user query using FAISS.
    Expects:
        - query: The search query string.
        - top_k: (Optional) Number of top summaries to return. Default is 4.
    Returns:
        - JSON containing the list of relevant summaries.
    """
    logger.debug("Received request to /search-summaries endpoint.")
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data provided in the request.")
            return jsonify({'error': 'No data provided.'}), 400

        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 4))  # Default to top 4 summaries

        if not query:
            logger.error("Empty query provided.")
            return jsonify({'error': 'Query cannot be empty.'}), 400

        logger.debug(f"Processing search query: '{query}' with top_k={top_k}")

        # Fetch relevant summaries using FAISS
        summaries = fetch_relevant_summaries(query, top_k)

        if not summaries:
            logger.info("No relevant summaries found.")
            return jsonify({'summaries': []}), 200

        logger.debug(f"Found {len(summaries)} relevant summaries.")
        return jsonify({'summaries': summaries}), 200

    except ValueError:
        logger.exception("Invalid top_k value provided.")
        return jsonify({'error': 'Invalid top_k value. It must be an integer.'}), 400
    except Exception as e:
        logger.exception(f"An error occurred while searching summaries: {e}")
        return jsonify({'error': 'Internal server error.'}), 500


def process_structured_journal_entry(journal_entry: str) -> Dict[str, Any]:
    logger.debug("Processing structured journal entry.")
    sections = journal_entry.split("\n\n")
    analysis_results = []

    # Analyze each section
    for section in sections:
        result = analyze_journal_entry(section)
        if 'error' in result:
            logger.error(f"Error in section analysis: {result['error']}")
            continue  # Skip sections with errors
        analysis_results.append(result)

    if not analysis_results:
        logger.error("No valid analysis results from sections.")
        return {
            'detected_emotions': [{'emotion': 'neutral', 'percentage': 100}],
            'detected_emotion': 'neutral',
            'suggested_action': "No suggested action available.",
            'affirmation': "Keep going, you are doing great!",
            'follow_up_prompt': "What would you like to reflect on next?",
            'context': {}
        }

    # Combine emotions
    combined_emotions = []
    combined_suggested_actions = []
    combined_affirmations = []
    combined_follow_up_prompts = []

    for result in analysis_results:
        combined_emotions.extend(result.get('detected_emotions', []))
        suggested_action = result.get('suggested_action')
        if suggested_action:
            combined_suggested_actions.append(suggested_action)
        affirmation = result.get('affirmation')
        if affirmation:
            combined_affirmations.append(affirmation)
        follow_up_prompt = result.get('follow_up_prompt')
        if follow_up_prompt:
            combined_follow_up_prompts.append(follow_up_prompt)

    # Aggregate emotions
    emotion_dict = {}
    for emotion in combined_emotions:
        name = emotion.get('emotion', 'neutral').lower()
        percentage = emotion.get('percentage', 0)
        emotion_dict[name] = emotion_dict.get(name, 0) + percentage

    # Normalize percentages to ensure they total 100%
    total_percentage = sum(emotion_dict.values())

    if total_percentage > 0:
        normalized_emotions = {
            emotion: (count / total_percentage) * 100 for emotion, count in emotion_dict.items()
        }
    else:
        normalized_emotions = emotion_dict  # Keep original if no emotions detected

    # Convert the aggregated emotions back to a list of Emotion objects
    mapped_emotions = [
        Emotion(emotion=emotion.title(), percentage=round(percentage, 2))  # Round to 2 decimal places
        for emotion, percentage in normalized_emotions.items()
        if percentage > 0  # Filter out emotions with 0% percentage
    ]

    # Convert Emotion objects to dictionaries
    mapped_emotions_dict = [emotion.dict() for emotion in mapped_emotions]

    # Aggregate affirmations and follow-up prompts
    # Here, we'll concatenate them. Modify as needed.
    aggregated_affirmation = " ".join(combined_affirmations) if combined_affirmations else "Keep going, you are doing great!"
    aggregated_follow_up_prompt = " ".join(combined_follow_up_prompts) if combined_follow_up_prompts else "What else would you like to explore?"

    # Determine the primary emotion (highest percentage)
    primary_emotion = mapped_emotions_dict[0]['emotion'].lower() if mapped_emotions_dict else 'neutral'

    # Aggregate suggested actions
    final_suggested_action = " ".join(combined_suggested_actions) if combined_suggested_actions else "No suggested action available."

    # Optionally, aggregate context if needed
    # For simplicity, we'll use the context from the first valid analysis result
    context = analysis_results[0].get('context', {})

    return {
        'detected_emotions': mapped_emotions_dict,  # Now a list of dicts
        'detected_emotion': primary_emotion,
        'suggested_action': final_suggested_action,
        'affirmation': aggregated_affirmation,
        'follow_up_prompt': aggregated_follow_up_prompt,
        'context': context
    }

# Handle Affirmations
def handle_affirmations(user_preferences: Dict[str, Any], analysis_result: Dict[str, Any]) -> None:
    affirmation = analysis_result.get('affirmation', "You are doing great!")  # Default value if no affirmation is provided
    analysis_result['affirmation'] = affirmation

# New Endpoint for Deleting Entries
@app.route('/delete-entry', methods=['DELETE'])
def delete_entry():
    """
    Endpoint to mark a journal entry as deleted (soft deletion).
    Expects:
        - user_id: ID of the user.
        - id: ID of the journal entry to delete.
    Returns:
        - Success or error message.
    """
    data = request.get_json()
    entry_id = data.get('id')  # Changed from 'entry_id' to 'id'
    user_id = data.get('user_id')

    if not entry_id or not user_id:
        logger.error("id and user_id are required.")
        return jsonify({'error': 'id and user_id are required.'}), 400

    try:
        if mark_entry_as_deleted(user_id, entry_id):  # Added user_id parameter
            logger.info(f"Entry {entry_id} marked as deleted successfully for user {user_id}.")
            return jsonify({'success': True}), 200
        else:
            logger.error(f"Entry {entry_id} does not exist or cannot be marked as deleted for user {user_id}.")
            return jsonify({'error': 'Entry not found or already deleted'}), 404
    except Exception as e:
        logger.exception(f"An error occurred while deleting entry {entry_id}: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# New Endpoint for Updating Entries
@app.route('/update-entry', methods=['PUT'])
def update_entry():
    """
    Endpoint to update specific fields of a journal entry.
    Expects:
        - user_id: ID of the user.
        - id: ID of the journal entry to update.
        - title: (Optional) New title.
        - journal_text: (Optional) Updated journal text.
        - font_style: (Optional) Updated font style.
    Returns:
        - Success or error message.
    """
    data = request.get_json()
    entry_id = data.get('id')  # Changed from 'entry_id' to 'id'
    user_id = data.get('user_id')  # Added user_id extraction
    title = data.get('title')
    journal_text = data.get('journal_text')
    font_style = data.get('font_style')

    if not entry_id or not user_id:
        logger.error("Entry ID and user_id are required.")
        return jsonify({'error': 'Entry ID and user_id are required'}), 400

    try:
        update_data = {k: v for k, v in {'title': title, 'journal_text': journal_text, 'font_style': font_style}.items() if v}

        if not update_data:
            logger.error("No data provided to update.")
            return jsonify({'error': 'No data provided to update.'}), 400

        success = update_journal_entry(user_id, entry_id, update_data)  # Added user_id parameter
        if not success:
            logger.error(f"Failed to update entry {entry_id} for user {user_id}.")
            return jsonify({'error': 'Failed to update entry.'}), 500

        logger.info(f"Entry {entry_id} updated successfully for user {user_id}.")
        return jsonify({'message': 'Entry updated successfully'}), 200

    except Exception as e:
        logger.exception(f"An error occurred while updating entry {entry_id}: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# New Endpoint for Recovering Deleted Entries
@app.route('/recover-entry', methods=['POST'])
def recover_entry():
    """
    Endpoint to recover a soft-deleted journal entry.
    Expects:
        - user_id: ID of the user.
        - id: ID of the journal entry to recover.
    Returns:
        - Success or error message.
    """
    data = request.get_json()
    entry_id = data.get('id')  # Changed from 'entry_id' to 'id'
    user_id = data.get('user_id')  # Added user_id extraction

    if not entry_id or not user_id:
        logger.error("Entry ID and user_id are required.")
        return jsonify({'error': 'Entry ID and user_id are required'}), 400

    try:
        success = recover_journal_entry(user_id, entry_id)  # Added user_id parameter
        if not success:
            logger.error(f"Failed to recover entry {entry_id} for user {user_id}. It may be past the recovery window.")
            return jsonify({'error': 'Failed to recover entry. It may be past the recovery window.'}), 500

        logger.info(f"Entry {entry_id} recovered successfully for user {user_id}.")
        return jsonify({'message': 'Entry recovered successfully'}), 200
    except Exception as e:
        logger.exception(f"An error occurred while recovering entry {entry_id}: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Automatic Permanent Deletion Logic
def permanently_delete_old_entries_task():
    """
    Scheduled task to permanently delete old entries that have been soft-deleted for over 15 days.
    """
    logger.debug("Running scheduled task to permanently delete old entries.")
    try:
        permanently_delete_old_entries()
        logger.debug("Successfully deleted old entries.")
    except Exception as e:
        logger.exception(f"Failed to delete old entries: {e}")


#       New Image Upload Endpoint #

@app.route('/upload-image', methods=['POST'])
def upload_image():
    """
    Endpoint to handle image uploads.
    Expects:
        - user_id: ID of the user uploading the image.
        - image: The image file to be uploaded.
    Returns:
        - image_url: The public URL of the uploaded image.
    """
    data = request.form
    user_id = data.get('user_id')

    if 'image' not in request.files:
        logger.error("No image file provided in the request.")
        return jsonify({'error': 'No image file provided.'}), 400

    image = request.files['image']

    if not user_id:
        logger.error("user_id is required.")
        return jsonify({'error': 'user_id is required.'}), 400

    if image.filename == '':
        logger.error("No selected file.")
        return jsonify({'error': 'No selected file.'}), 400

    # Validate image type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    if not ('.' in image.filename and image.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        logger.error("Invalid image file type.")
        return jsonify({'error': 'Invalid image file type.'}), 400

    # Validate file size (e.g., max 5MB)
    image.seek(0, os.SEEK_END)
    file_size = image.tell()
    max_size = 5 * 1024 * 1024  # 5MB
    if file_size > max_size:
        logger.error(f"Image file size exceeds the limit: {file_size} bytes.")
        return jsonify({'error': 'Image file size exceeds the 5MB limit.'}), 400
    image.seek(0)  # Reset file pointer

    try:
        # Generate a unique filename
        file_extension = image.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"

        # Define the storage path
        from firebase_admin import storage
        bucket = storage.bucket()
        blob = bucket.blob(f"journal_images/{user_id}/{unique_filename}")

        # Upload the image to Firebase Storage
        blob.upload_from_file(image, content_type=image.content_type)
        blob.make_public()  # Make the blob publicly viewable

        image_url = blob.public_url
        logger.info(f"Image uploaded successfully for user {user_id}. URL: {image_url}")
        return jsonify({'image_url': image_url}), 200

    except Exception as e:
        logger.exception(f"An error occurred while uploading image: {e}")
        return jsonify({'error': 'Failed to upload image.'}), 500


#    Scraper Integration      #


# Import the MentalHealthSpider and related dependencies at the top of app.py
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor
import threading
from scrapy import signals  # Updated import
from scrapy.signalmanager import dispatcher
from datetime import datetime
from scrapy_project.scrapy_project.spiders.mental_health_spider import MentalHealthSpider  # Import the spider we created

# Global variable to track scraping status
scraping_status = {
    'is_running': False,
    'start_time': None,
    'summaries_scraped': 0,  # Changed from 'articles_scraped' to 'summaries_scraped'
    'last_error': None,
    'completed': False
}

def spider_closed(spider):
    """Callback function when spider closes."""
    global scraping_status
    scraping_status['is_running'] = False
    scraping_status['completed'] = True
    logger.info("Spider finished scraping")

def spider_error(failure, response, spider):
    """Callback function for spider errors."""
    global scraping_status
    scraping_status['last_error'] = str(failure)
    logger.error(f"Spider error: {failure}")

def item_scraped(item, response, spider):
    """Callback function when an item is scraped."""
    global scraping_status
    scraping_status['summaries_scraped'] += 1  # Changed from 'articles_scraped' to 'summaries_scraped'
    logger.debug(f"Scraped article: {item.get('title', 'No title')}")

    # Process the scraped item: summarize and save to Firestore
    try:
        article_content = item.get('content', '')
        article_id = item.get('id', str(uuid.uuid4()))  # Generate a unique ID if not provided
        if not article_content:
            logger.warning(f"Scraped article '{item.get('title', 'No title')}' has no content. Skipping.")
            return

        # Summarize the article using GPTSummarizer
        summarized_data = summarizer.summarize_article(article_content)
        if summarized_data and 'error' not in summarized_data:
            # Generate a sanitized article ID
            sanitized_id = sanitize_article_id(article_id)
            # Add the summarized article to Firestore
            success = summarizer.add_article(sanitized_id, summarized_data)
            if success:
                logger.info(f"Summarized article '{item.get('title', 'No title')}' saved successfully.")
            else:
                logger.error(f"Failed to save summarized article '{item.get('title', 'No title')}'.")
        else:
            logger.error(f"Summarization failed for article '{item.get('title', 'No title')}': {summarized_data.get('error', 'Unknown error')}")

    except Exception as e:
        logger.exception(f"Error processing scraped item: {e}")

def run_spider_in_thread():
    """Run the spider in a separate thread."""
    def _run_spider():
        try:
            # Configure the crawler
            settings = get_project_settings()
            settings.set('LOG_ENABLED', True)
            settings.set('LOG_LEVEL', 'INFO')
            
            # Initialize the runner
            runner = CrawlerRunner(settings)
            
            # Register callbacks
            dispatcher.connect(spider_closed, signal=signals.spider_closed)
            dispatcher.connect(spider_error, signal=signals.spider_error)
            dispatcher.connect(item_scraped, signal=signals.item_scraped)
            
            # Start the crawler
            deferred = runner.crawl(MentalHealthSpider)
            deferred.addBoth(lambda _: reactor.stop())
            
            # Run the reactor
            reactor.run(installSignalHandlers=False)
            
        except Exception as e:
            global scraping_status
            scraping_status['last_error'] = str(e)
            scraping_status['is_running'] = False
            logger.exception("Error running spider")

    # Start the spider in a new thread
    thread = threading.Thread(target=_run_spider)
    thread.start()

@app.route('/start-scraper', methods=['POST'])
def start_scraper():
    """
    Endpoint to start the article scraping process.
    Returns:
        JSON response with scraping status.
    """
    global scraping_status
    
    try:
        # Check if scraper is already running
        if scraping_status['is_running']:
            return jsonify({
                'status': 'error',
                'message': 'Scraper is already running',
                'started_at': scraping_status['start_time'].isoformat() if scraping_status['start_time'] else None
            }), 409

        # Reset status
        scraping_status = {
            'is_running': True,
            'start_time': datetime.now(timezone.utc),
            'summaries_scraped': 0,  # Changed from 'articles_scraped' to 'summaries_scraped'
            'last_error': None,
            'completed': False
        }

        # Start the spider in a separate thread
        run_spider_in_thread()

        return jsonify({
            'status': 'success',
            'message': 'Article scraping started successfully',
            'started_at': scraping_status['start_time'].isoformat()
        }), 200

    except Exception as e:
        logger.exception("Error starting scraper")
        scraping_status['is_running'] = False
        scraping_status['last_error'] = str(e)
        return jsonify({
            'status': 'error',
            'message': f'Failed to start scraper: {str(e)}'
        }), 500

@app.route('/scraper-status', methods=['GET'])
def get_scraper_status():
    """
    Endpoint to check the status of the article scraping process.
    Returns:
        JSON response with current scraping status.
    """
    global scraping_status
    
    try:
        status_response = {
            'is_running': scraping_status['is_running'],
            'summaries_scraped': scraping_status['summaries_scraped'],  # Changed from 'articles_scraped'
            'completed': scraping_status['completed'],
            'start_time': scraping_status['start_time'].isoformat() if scraping_status['start_time'] else None,
            'last_error': scraping_status['last_error']
        }

        if scraping_status['completed']:
            status_response['message'] = 'Scraping completed successfully'
        elif scraping_status['is_running']:
            status_response['message'] = 'Scraping in progress'
        else:
            status_response['message'] = 'Scraper is not running'

        return jsonify(status_response), 200

    except Exception as e:
        logger.exception("Error getting scraper status")
        return jsonify({
            'status': 'error',
            'message': f'Failed to get scraper status: {str(e)}'
        }), 500

@app.route('/stop-scraper', methods=['POST'])
def stop_scraper():
    """
    Endpoint to stop the article scraping process.
    Returns:
        JSON response confirming scraper stoppage.
    """
    global scraping_status
    
    try:
        if not scraping_status['is_running']:
            return jsonify({
                'status': 'error',
                'message': 'Scraper is not running'
            }), 409

        # Stop the reactor if it's running
        if reactor.running:
            reactor.callFromThread(reactor.stop)

        scraping_status['is_running'] = False
        scraping_status['completed'] = True

        return jsonify({
            'status': 'success',
            'message': 'Scraper stopped successfully',
            'summaries_scraped': scraping_status['summaries_scraped']  # Changed from 'articles_scraped'
        }), 200

    except Exception as e:
        logger.exception("Error stopping scraper")
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop scraper: {str(e)}'
        }), 500

@app.route('/process-response', methods=['POST']) 
def process_response():
    """
    Endpoint to process the user's journal entry and provide emotional analysis, suggestions, and relevant summaries.
    """
    logger.debug("Processing /process-response request.")
    try:
        # Validate request data
        data = request.get_json()
        if not data:
            logger.error("No JSON data in request")
            return jsonify({
                "error": "No data provided",
                "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                "detected_emotion": 'neutral',
                "suggested_action": "Please provide journal content to analyze.",
                "affirmation": "We're here when you're ready to share your thoughts.",
                "follow_up_prompt": "Would you like to try writing something?",
                "context": {},
                "keywords": [],
                "theme": None,
                "recommended_summaries": []
            }), 200  # Return 200 with default values

        user_id = data.get('user_id')
        journal_entry = data.get('journal_entry')
        journaling_style = data.get('journaling_style', 'freeform')
        image_urls = data.get('images', [])

        # Validate required fields
        if not user_id or not journal_entry:
            logger.error("Missing user_id or journal_entry in the request.")
            return jsonify({
                "error": "Missing required fields",
                "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                "detected_emotion": 'neutral',
                "suggested_action": "Please ensure all required information is provided.",
                "affirmation": "Taking time to journal is valuable - let's make sure we have everything needed.",
                "follow_up_prompt": "Would you like to try again with all required information?",
                "context": {},
                "keywords": [],
                "theme": None,
                "recommended_summaries": []
            }), 200  # Return 200 with default values

        # Fetch user preferences
        try:
            user_preferences = get_user_preferences(user_id)
            if not user_preferences:
                logger.error(f"User with ID {user_id} does not exist.")
                return jsonify({
                    "error": "User not found",
                    "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                    "detected_emotion": 'neutral',
                    "suggested_action": "Please ensure you're logged in correctly.",
                    "affirmation": "Your identity is important to us.",
                    "follow_up_prompt": "Would you like to try logging in again?",
                    "context": {},
                    "keywords": [],
                    "theme": None,
                    "recommended_summaries": []
                }), 200  # Return 200 with default values
        except Exception as e:
            logger.error(f"Error fetching user preferences: {str(e)}")
            user_preferences = {}

        # Process journal entry based on style
        if journaling_style == 'structured' and isinstance(journal_entry, dict):
            entry_text = " ".join(f"{key}: {value}" for key, value in journal_entry.items() if value)
        elif isinstance(journal_entry, str):
            entry_text = journal_entry
        else:
            logger.error("Invalid journal entry format.")
            return jsonify({
                "error": "Invalid format",
                "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                "detected_emotion": 'neutral',
                "suggested_action": "The journal entry format wasn't quite right.",
                "affirmation": "Let's try again with your thoughts in text form.",
                "follow_up_prompt": "Would you like to share your thoughts in a different way?",
                "context": {},
                "keywords": [],
                "theme": None,
                "recommended_summaries": []
            }), 200  # Return 200 with default values

        if not entry_text.strip():
            logger.error("Journal entry is empty.")
            return jsonify({
                "error": "Empty entry",
                "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                "detected_emotion": 'neutral',
                "suggested_action": "Try writing down what's on your mind.",
                "affirmation": "Even a few words can be meaningful.",
                "follow_up_prompt": "What's the first thing that comes to mind?",
                "context": {},
                "keywords": [],
                "theme": None,
                "recommended_summaries": []
            }), 200  # Return 200 with default values

        # Analyze the journal entry
        try:
            if journaling_style == 'structured':
                analysis_result = process_structured_journal_entry(entry_text)
            else:
                analysis_result = analyze_journal_entry(entry_text)
        except Exception as e:
            logger.error(f"Error analyzing journal entry: {str(e)}")
            return jsonify({
                "error": "Analysis error",
                "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
                "detected_emotion": 'neutral',
                "suggested_action": "We had trouble analyzing your entry, but your thoughts are still valuable.",
                "affirmation": "Technical issues happen, but your practice matters most.",
                "follow_up_prompt": "Would you like to try again?",
                "context": {},
                "keywords": [],
                "theme": None,
                "recommended_summaries": []
            }), 200  # Return 200 with default values

        # Format detected emotions
        if not isinstance(analysis_result.get('detected_emotions'), list):
            if isinstance(analysis_result.get('detected_emotions'), dict):
                analysis_result['detected_emotions'] = [
                    {'emotion': emotion, 'percentage': float(percentage.strip('%'))}
                    for emotion, percentage in analysis_result['detected_emotions'].items()
                ]
            else:
                analysis_result['detected_emotions'] = [{'emotion': 'neutral', 'percentage': 100}]

        # Extract keywords, theme, and recommended summaries
        try:
            # Extract keywords from detected emotions
            keywords = [emotion['emotion'].lower() for emotion in analysis_result['detected_emotions']]

            # Extract keywords from context
            context = analysis_result.get('context', {})
            if context:
                keywords.extend([word.lower() for word in context.get('people_mentioned', [])])
                keywords.extend([word.lower() for word in context.get('places_mentioned', [])])
                keywords.extend([word.lower() for word in context.get('events', [])])

            # Remove duplicate keywords
            keywords = list(set(keywords))
            logger.debug(f"Extracted keywords: {keywords}")

            # Fetch synonyms for each keyword
            synonyms = {keyword: get_synonyms(keyword) for keyword in keywords}
            logger.debug(f"Fetched synonyms: {synonyms}")

            # Fetch theme
            theme = analysis_result.get('theme')
            if theme:
                theme_synonyms = get_synonyms(theme)
                logger.debug(f"Fetched synonyms for theme '{theme}': {theme_synonyms}")
            else:
                logger.debug("No theme detected.")

            # Calculate user sentiment
            user_sentiment = calculate_user_sentiment(analysis_result)
            logger.debug(f"Calculated user sentiment: {user_sentiment}")

            # Fetch recommended summaries
            recommended_summaries = fetch_relevant_summaries(
                journal_entry=entry_text,
                user_sentiment=user_sentiment
            )
            logger.debug(f"Recommended summaries: {recommended_summaries}")

        except Exception as e:
            logger.error(f"Error processing keywords, theme, and summaries: {str(e)}")
            keywords = []
            theme = None
            recommended_summaries = []

        # Prepare success response
        response_data = {
            "detected_emotions": analysis_result['detected_emotions'],
            "detected_emotion": analysis_result['detected_emotions'][0]['emotion'] if analysis_result['detected_emotions'] else 'neutral',
            "suggested_action": analysis_result.get('suggested_action', 'Take a moment to reflect on your thoughts.'),
            "affirmation": analysis_result.get('affirmation', 'Your insights are valuable.'),
            "follow_up_prompt": analysis_result.get('follow_up_prompt', 'What else would you like to explore?'),
            "context": analysis_result.get('context', {
                "people_mentioned": [],
                "places_mentioned": [],
                "events": []
            }),
            "keywords": keywords,
            "theme": theme,
            "recommended_summaries": recommended_summaries
        }

        logger.debug("Successfully processed journal entry and prepared response.")
        return jsonify(response_data), 200

    except Exception as e:
        logger.exception(f"Unexpected error in process_response: {str(e)}")
        return jsonify({
            "error": "Unexpected error",
            "detected_emotions": [{'emotion': 'neutral', 'percentage': 100}],
            "detected_emotion": 'neutral',
            "suggested_action": "We encountered an unexpected issue, but your journaling practice is what matters most.",
            "affirmation": "Technical difficulties can't diminish the value of your reflections.",
            "follow_up_prompt": "Would you like to try again?",
            "context": {},
            "keywords": [],
            "theme": None,
            "recommended_summaries": []
        }), 200  # Return 200 even for unexpected errors


@app.route('/save-emotional-baseline', methods=['POST'])
def save_emotional_baseline():
    """
    Endpoint to save the user's emotional baseline preferences.
    Expects:
        - user_id: ID of the user.
        - currentMood: The user's current mood.
        - selectedEmotions: List of selected emotions.
    Returns:
        - Success or error message.
    """
    logger.debug("Received request for /save-emotional-baseline")

    try:
        data = request.get_json()
        user_id = data.get('user_id')
        current_mood = data.get('currentMood')
        selected_emotions = data.get('selectedEmotions')

        if not user_id or current_mood is None or not selected_emotions:
            logger.error("Missing user ID, current mood, or selected emotions data.")
            return jsonify({"error": "Missing user ID, current mood, or selected emotions data."}), 400

        success = update_user_preferences(user_id, {
            'emotional_baseline': {
                'current_mood': current_mood,
                'selected_emotions': selected_emotions
            }
        })
        if not success:
            logger.error("Failed to save emotional baseline.")
            return jsonify({"error": "Failed to save emotional baseline."}), 500

        logger.debug(f"Saved emotional baseline for user {user_id}")
        return jsonify({"message": "Emotional baseline saved successfully."}), 200

    except Exception as e:
        logger.exception(f"An error occurred while saving emotional baseline: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

@app.route('/get-themes', methods=['GET'])
def get_themes():
    """
    Retrieves all available journaling themes.
    """
    try:
        themes = [
            {
                'name': 'icebreaker_prompts',
                'display_name': 'Icebreaker Prompts',
                'description': 'General prompts to help you get started with journaling.'
            },
            {
                'name': 'adhd_management',
                'display_name': 'ADHD Management',
                'description': 'Prompts to help manage ADHD symptoms.'
            },
            {
                'name': 'gratitude_journal',
                'display_name': 'Gratitude Journal',
                'description': 'Prompts to cultivate gratitude and appreciation.'
            },
            {
                'name': 'anxiety_management',
                'display_name': 'Anxiety Management',
                'description': 'Prompts to help manage anxiety.'
            },
            {
                'name': 'self_compassion',
                'display_name': 'Self-Compassion',
                'description': 'Prompts to practice self-compassion.'
            },
            {
                'name': 'motivation_goal_setting',
                'display_name': 'Motivation & Goal Setting',
                'description': 'Prompts to boost motivation and set goals.'
            },
            {
                'name': 'sleep_quality_tracker',
                'display_name': 'Sleep Quality Tracker',
                'description': 'Prompts to track and improve sleep quality.'
            },
            {
                'name': 'mindfulness_practice',
                'display_name': 'Mindfulness Practice',
                'description': 'Prompts to enhance mindfulness.'
            },
            {
                'name': 'relationship_building',
                'display_name': 'Relationship Building',
                'description': 'Prompts to improve relationships.'
            },
            {
                'name': 'personal_growth',
                'display_name': 'Personal Growth',
                'description': 'Prompts for personal development.'
            },
            {
                'name': 'creativity_inspiration',
                'display_name': 'Creativity & Inspiration',
                'description': 'Prompts to spark creativity.'
            },
            # Add other themes as needed
        ]
        logger.debug(f"Retrieved themes: {themes}")
        return jsonify({'themes': themes}), 200
    except Exception as e:
        logger.exception(f"An error occurred while retrieving themes: {e}")
        return jsonify({'error': 'Failed to retrieve themes.'}), 500

@app.route('/prompts/<theme_name>', methods=['GET'])
def get_prompts(theme_name):
    """
    Retrieves prompts based on the selected theme.
    """
    # Map theme names to their respective prompt generation functions
    theme_functions = {
        'icebreaker_prompts': generate_icebreaker_prompts,
        'adhd_management': generate_adhd_management_prompts,
        'gratitude_journal': generate_gratitude_journal_prompts,
        'anxiety_management': generate_anxiety_management_prompts,
        'self_compassion': generate_self_compassion_prompts,
        'motivation_goal_setting': generate_motivation_goal_setting_prompts,
        'sleep_quality_tracker': generate_sleep_quality_tracker_prompts,
        'mindfulness_practice': generate_mindfulness_practice_prompts,
        'relationship_building': generate_relationship_building_prompts,
        'personal_growth': generate_personal_growth_prompts,
        'creativity_inspiration': generate_creativity_inspiration_prompts,
        # Add other themes as needed
    }

    generate_prompts_function = theme_functions.get(theme_name)

    if not generate_prompts_function:
        logger.error(f"No prompt generation function found for theme: {theme_name}")
        return jsonify({'error': f'Invalid theme name: {theme_name}'}), 400

    try:
        prompts = generate_prompts_function()
        if prompts:
            # Assuming prompts are objects with a 'prompt' attribute
            formatted_prompts = [p.prompt for p in prompts]
            logger.debug(f"Generated prompts for theme '{theme_name}': {formatted_prompts}")
            return jsonify({"prompts": formatted_prompts}), 200
        else:
            logger.error(f"Failed to generate prompts for theme: {theme_name}")
            return jsonify({"error": f"Failed to generate prompts for {theme_name}."}), 500
    except Exception as e:
        logger.exception(f"An error occurred while generating prompts for {theme_name}: {e}")
        return jsonify({"error": f"An error occurred while generating prompts for {theme_name}."}), 500

# Main Entry Point
if __name__ == '__main__':
    try:
        app.run(debug=True, host='localhost', port=5000)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down successfully.")
