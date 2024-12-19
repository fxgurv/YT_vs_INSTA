from flask import Flask, request, jsonify, send_file
import os
import json
import psutil
import datetime
import platform
import schedule
import threading
import uuid
import time
from classes.YouTube import YouTube
from cache import get_accounts, add_account as cache_add_account, remove_account as cache_remove_account
from status import info, success, error
from utils import close_running_selenium_instances, rem_temp_files
from config import assert_folder_structure

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('Index.html')

@app.route('/api/system-info')
def system_info():
    # Get system information
    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.datetime.now() - boot_time
    
    # Get storage information
    disk = psutil.disk_usage('/')
    storage = f"Total: {bytes_to_gb(disk.total)}GB, Used: {bytes_to_gb(disk.used)}GB, Free: {bytes_to_gb(disk.free)}GB"
    
    return jsonify({
        'os': f"{platform.system()} {platform.release()}",
        'uptime': str(uptime).split('.')[0],
        'storage': storage
    })

@app.route('/api/accounts')
def get_youtube_accounts():
    try:
        accounts = get_accounts("youtube")
        return jsonify(accounts)
    except Exception as e:
        error(f"Error fetching accounts: {str(e)}")
        return jsonify({'error': 'Failed to fetch accounts'}), 500

@app.route('/api/account', methods=['POST'])
def add_account():
    try:
        data = request.json
        account = {
            'id': str(uuid.uuid4()),
            'name': data['name'],
            'profile_path': data['profile_path'],
            'niche': data['niche'],
            'language': data['language']
        }
        cache_add_account("youtube", account)
        success(f"Added account: {account['name']}")
        return jsonify({'message': 'Account added successfully', 'account': account}), 201
    except Exception as e:
        error(f"Error adding account: {str(e)}")
        return jsonify({'error': 'Failed to add account'}), 500

@app.route('/api/account/<id>', methods=['DELETE'])
def remove_account(id):
    try:
        cache_remove_account(id)
        success(f"Removed account with ID: {id}")
        return jsonify({'message': 'Account removed successfully'})
    except Exception as e:
        error(f"Error removing account: {str(e)}")
        return jsonify({'error': 'Failed to remove account'}), 500

@app.route('/api/generate/<id>')
def generate_video(id):
    try:
        accounts = get_accounts("youtube")
        account = next((acc for acc in accounts if acc['id'] == id), None)
        if not account:
            return jsonify({'error': 'Account not found'}), 404
        
        # Initialize YouTube class with account details
        yt = YouTube(
            account['id'],
            account['name'],
            account['profile_path'],
            account['niche'],
            account['language']
        )
        
        # Generate and upload video
        video_path = yt.generate_video()
        if video_path:
            success(f"Generated video for account: {account['name']}")
            # Start upload in a separate thread
            threading.Thread(target=yt.upload_video).start()
            return jsonify({'message': 'Video generation completed and upload started'})
        else:
            return jsonify({'error': 'Video generation failed'}), 500
    except Exception as e:
        error(f"Error generating video: {str(e)}")
        return jsonify({'error': 'Failed to generate video'}), 500

@app.route('/api/cron/<id>', methods=['POST'])
def setup_cron(id):
    try:
        data = request.json
        schedule_type = data.get('schedule_type')
        
        accounts = get_accounts("youtube")
        account = next((acc for acc in accounts if acc['id'] == id), None)
        if not account:
            return jsonify({'error': 'Account not found'}), 404
        
        # Schedule video generation based on schedule_type
        if schedule_type == 'daily':
            schedule.every().day.at("10:00").do(
                lambda: generate_video_task(account)
            )
        elif schedule_type == 'weekly':
            schedule.every().monday.at("10:00").do(
                lambda: generate_video_task(account)
            )
        
        success(f"Set {schedule_type} schedule for account: {account['name']}")
        return jsonify({'message': f'Schedule set to {schedule_type}'})
    except Exception as e:
        error(f"Error setting schedule: {str(e)}")
        return jsonify({'error': 'Failed to set schedule'}), 500

def generate_video_task(account):
    """Helper function for scheduled video generation"""
    try:
        yt = YouTube(
            account['id'],
            account['name'],
            account['profile_path'],
            account['niche'],
            account['language']
        )
        video_path = yt.generate_video()
        if video_path:
            yt.upload_video()
    except Exception as e:
        error(f"Scheduled task failed: {str(e)}")

def bytes_to_gb(bytes):
    return round(bytes / (1024**3), 2)

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

def initialize():
    """Initialize application requirements"""
    try:
        # Ensure folder structure exists
        assert_folder_structure()
        # Clean temporary files
        rem_temp_files()
        # Close any running selenium instances
        close_running_selenium_instances()
        success("Initialization completed successfully")
    except Exception as e:
        error(f"Initialization failed: {str(e)}")

if __name__ == '__main__':
    # Initialize application
    initialize()
    
    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_schedule)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=7860)
