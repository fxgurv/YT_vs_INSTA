import os
import json
import psutil
import schedule
import subprocess
import threading
from art import *
from cache import *
from utils import *
from config import *
from status import *
from uuid import uuid4
from constants import *
from termcolor import colored
from classes.YouTube import YouTube
from prettytable import PrettyTable
from flask import Flask, jsonify, request
from werkzeug.serving import make_server



app = Flask(__name__)


@app.route('/')
def home():
    with open('Index.html', 'r') as f:
        return f.read()

@app.route('/api/system-info')
def api_system_info():
    info = get_system_info()
    return jsonify(info)

@app.route('/api/accounts')
def get_youtube_accounts():
    accounts = get_accounts("youtube")
    return jsonify(accounts)

@app.route('/api/account', methods=['POST'])
def add_youtube_account():
    data = request.json
    new_account = {
        "id": str(uuid4()),
        "name": data['name'],
        "firefox_profile": data['firefox_profile'],
        "niche": data['niche'],
        "language": data['language'],
        "videos": []
    }
    add_account("youtube", new_account)
    return jsonify({"status": "success", "account": new_account})

@app.route('/api/account/<id>', methods=['DELETE'])
def remove_youtube_account(id):
    remove_account("youtube", id)
    return jsonify({"status": "success"})

@app.route('/api/generate/<id>')
def generate_video(id):
    accounts = get_accounts("youtube")
    account = next((acc for acc in accounts if acc["id"] == id), None)
    if account:
        youtube = YouTube(
            account["id"],
            account["name"],
            account["firefox_profile"],
            account["niche"],
            account["language"]
        )
        youtube.generate_video()
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Account not found"})

@app.route('/api/cron/<id>', methods=['POST'])
def setup_cron(id):
    data = request.json
    schedule_type = data['schedule_type']
    accounts = get_accounts("youtube")
    account = next((acc for acc in accounts if acc["id"] == id), None)
    if account:
        cron_script_path = os.path.join(ROOT_DIR, "src", "cron.py")
        command = f"python {cron_script_path} youtube {account['id']}"
        
        def job():
            subprocess.run(command)
            
        if schedule_type == "daily":
            schedule.every(1).day.do(job)
        elif schedule_type == "twice":
            schedule.every().day.at("10:00").do(job)
            schedule.every().day.at("16:00").do(job)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Account not found"})


def main():
    info("Starting main function")
    valid_input = False
    while not valid_input:
        try:
            info("Displaying options to user")
            info("\n============ Main Menu ============", False)

            for idx, option in enumerate(MAIN_MENU):
                print(colored(f" {idx + 1}. {option}", "cyan"))

            info("=================================\n", False)
            user_input = input("Select an option: ").strip()
            if user_input == '':
                warning("Empty input received")
                print("\n" * 100)
                raise ValueError("Empty input is not allowed.")
            user_input = int(user_input)
            valid_input = True
            info(f"User selected option: {user_input}")
        except ValueError as e:
            error(f"Invalid input: {e}")
            print("\n" * 100)

    if user_input == 1:
        info("Starting Shorts Automater...")

        cached_accounts = get_accounts("youtube")
        info(f"Retrieved {len(cached_accounts)} cached Shorts accounts")

        if len(cached_accounts) == 0:
            warning("No accounts found in cache. Prompting to create one.")
            user_input = question("y/n: ")

            if user_input.lower() == "y":
                generated_uuid = str(uuid4())
                info(f"Generated new UUID: {generated_uuid}")

                success(f" => Generated ID: {generated_uuid}")
                name = question(" => Enter a name for this account: ")
                fp_profile = question(" => Enter the path to the Firefox profile: ")
                niche = question(" => Enter the account niche: ")
                language = question(" => Enter the account language: ")

                add_account("youtube", {
                    "id": generated_uuid,
                    "name": name,
                    "firefox_profile": fp_profile,
                    "niche": niche,
                    "language": language,
                    "videos": []
                })
                success(f"Added new Shorts account: {name}")
        else:
            info("Displaying cached Shorts accounts")
            table = PrettyTable()
            table.field_names = ["ID", "UUID", "Nickname", "Niche"]

            for account in cached_accounts:
                table.add_row([cached_accounts.index(account) + 1, colored(account["id"], "cyan"), colored(account["name"], "blue"), colored(account["niche"], "green")])

            print(table)

            user_input = question("Select an account to start: ")

            selected_account = None

            for account in cached_accounts:
                if str(cached_accounts.index(account) + 1) == user_input:
                    selected_account = account

            if selected_account is None:
                error("Invalid account selected. Restarting main function.")
                main()
            else:
                info(f"Selected YouTube account: {selected_account['name']}")
                youtube = YouTube(
                    selected_account["id"],
                    selected_account["name"],
                    selected_account["firefox_profile"],
                    selected_account["niche"],
                    selected_account["language"]
                )

                while True:
                    rem_temp_files()
                    info("Removed temporary files")
                    info("\n============ SHORTS MENU ============", False)

                    for idx, youtube_option in enumerate(SHORTS_MENU):
                        print(colored(f" {idx + 1}. {youtube_option}", "cyan"))

                    info("=================================\n", False)

                    user_input = int(question("Select an option: "))
                    info(f"User selected option: {user_input}")

                    if user_input == 1:
                        info("Generating Short video")
                        youtube.generate_video()
                        upload_to_yt = question("Do you want to upload this video to YouTube? (y/n): ")
                        if upload_to_yt.lower() == "y":
                            info("Uploading video to YouTube")
                            youtube.upload_video()
                    elif user_input == 2:
                        info("Retrieving Short videos")
                        videos = youtube.get_videos()

                        if len(videos) > 0:
                            info(f"Displaying {len(videos)} videos")
                            videos_table = PrettyTable()
                            videos_table.field_names = ["ID", "Date", "Title"]

                            for video in videos:
                                videos_table.add_row([
                                    videos.index(video) + 1,
                                    colored(video["date"], "blue"),
                                    colored(video["title"][:60] + "...", "green")
                                ])

                            print(videos_table)
                        else:
                            warning("No videos found.")
                    elif user_input == 3:
                        info("Setting up CRON job for Shorts uploads")
                        info("How often do you want to upload?")

                        info("\n============ OPTIONS ============", False)
                        for idx, cron_option in enumerate(CRON_MENU):
                            print(colored(f" {idx + 1}. {cron_option}", "cyan"))

                        info("=================================\n", False)

                        user_input = int(question("Select an Option: "))

                        cron_script_path = os.path.join(ROOT_DIR, "src", "cron.py")
                        command = f"python {cron_script_path} youtube {selected_account['id']}"

                        def job():
                            info("Executing CRON job for Shorts upload")
                            subprocess.run(command)

                        if user_input == 1:
                            info("Setting up daily upload")
                            schedule.every(1).day.do(job)
                            success("Set up CRON Job for daily upload.")
                        elif user_input == 2:
                            info("Setting up twice daily upload")
                            schedule.every().day.at("10:00").do(job)
                            schedule.every().day.at("16:00").do(job)
                            success("Set up CRON Job for twice daily upload.")
                        else:
                            info("Returning to main menu")
                            break
                    elif user_input == 4:
                        if get_verbose():
                            info(" => Climbing Options Ladder...", False)
                        info("Returning to main menu")
                        break

    elif user_input == 2:
        info("Starting Twitter Bot...")

        cached_accounts = get_accounts("twitter")
        info(f"Retrieved {len(cached_accounts)} cached Twitter accounts")

        if len(cached_accounts) == 0:
            warning("No accounts found in cache. Prompting to create one.")
            user_input = question("Yes/No: ")

            if user_input.lower() == "yes":
                generated_uuid = str(uuid4())
                info(f"Generated new UUID: {generated_uuid}")

                success(f" => Generated ID: {generated_uuid}")
                name = question(" => Enter a name for this account: ")
                fp_profile = question(" => Enter the path to the Firefox profile: ")
                topic = question(" => Enter the account topic: ")

                add_account("twitter", {
                    "id": generated_uuid,
                    "name": name,
                    "firefox_profile": fp_profile,
                    "topic": topic,
                    "posts": []
                })
                success(f"Added new Twitter account: {name}")
        else:
            info("Displaying cached Twitter accounts")
            table = PrettyTable()
            table.field_names = ["ID", "UUID", "Nickname", "Account Topic"]

            for account in cached_accounts:
                table.add_row([cached_accounts.index(account) + 1, colored(account["id"], "cyan"), colored(account["name"], "blue"), colored(account["topic"], "green")])

            print(table)

            user_input = question("Select an account to start: ")

            selected_account = None

            for account in cached_accounts:
                if str(cached_accounts.index(account) + 1) == user_input:
                    selected_account = account

            if selected_account is None:
                error("Invalid account selected. Restarting main function.")
                main()
            else:
                info(f"Selected Twitter account: {selected_account['name']}")
                twitter = Twitter(
                    selected_account["id"],
                    selected_account["name"],
                    selected_account["firefox_profile"],
                    selected_account["topic"]
                )

                while True:
                    rem_temp_files()
                    info("Removed temporary files")
                    info("\n============ OPTIONS ============", False)

                    for idx, twitter_option in enumerate(TWITTER_OPTIONS):
                        print(colored(f" {idx + 1}. {twitter_option}", "cyan"))

                    info("=================================\n", False)

                    user_input = int(question("Select an option: "))
                    info(f"User selected Twitter option: {user_input}")

                    if user_input == 1:
                        info("Posting to Twitter")
                        twitter.post()
                    elif user_input == 2:
                        info("Retrieving Twitter posts")
                        posts = twitter.get_posts()

                        if len(posts) > 0:
                            info(f"Displaying {len(posts)} posts")
                            posts_table = PrettyTable()
                            posts_table.field_names = ["ID", "Date", "Content"]

                            for post in posts:
                                posts_table.add_row([
                                    posts.index(post) + 1,
                                    colored(post["date"], "blue"),
                                    colored(post["content"][:60] + "...", "green")
                                ])

                            print(posts_table)
                        else:
                            warning("No posts found.")
                    elif user_input == 3:
                        info("Setting up CRON job for Twitter posts")
                        info("How often do you want to post?")

                        info("\n============ OPTIONS ============", False)
                        for idx, cron_option in enumerate(TWITTER_CRON_OPTIONS):
                            print(colored(f" {idx + 1}. {cron_option}", "cyan"))

                        info("=================================\n", False)

                        user_input = int(question("Select an Option: "))

                        cron_script_path = os.path.join(ROOT_DIR, "src", "cron.py")
                        command = f"python {cron_script_path} twitter {selected_account['id']}"

                        def job():
                            info("Executing CRON job for Twitter post")
                            subprocess.run(command)

                        if user_input == 1:
                            info("Setting up daily post")
                            schedule.every(1).day.do(job)
                            success("Set up CRON Job for daily post.")
                        elif user_input == 2:
                            info("Setting up twice daily post")
                            schedule.every().day.at("10:00").do(job)
                            schedule.every().day.at("16:00").do(job)
                            success("Set up CRON Job for twice daily post.")
                        elif user_input == 3:
                            info("Setting up thrice daily post")
                            schedule.every().day.at("08:00").do(job)
                            schedule.every().day.at("12:00").do(job)
                            schedule.every().day.at("18:00").do(job)
                            success("Set up CRON Job for thrice daily post.")
                        else:
                            info("Returning to main menu")
                            break
                    elif user_input == 4:
                        if get_verbose():
                            info(" => Climbing Options Ladder...", False)
                        info("Returning to main menu")
                        break

    elif user_input == 3:
        info("Starting Affiliate Marketing...")

        cached_products = get_products()
        info(f"Retrieved {len(cached_products)} cached products")

        if len(cached_products) == 0:
            warning("No products found in cache. Prompting to create one.")
            user_input = question("Yes/No: ")

            if user_input.lower() == "yes":
                affiliate_link = question(" => Enter the affiliate link: ")
                twitter_uuid = question(" => Enter the Twitter Account UUID: ")

                # Find the account
                account = None
                for acc in get_accounts("twitter"):
                    if acc["id"] == twitter_uuid:
                        account = acc

                add_product({
                    "id": str(uuid4()),
                    "affiliate_link": affiliate_link,
                    "twitter_uuid": twitter_uuid
                })
                success(f"Added new product with affiliate link: {affiliate_link}")

                afm = AffiliateMarketing(affiliate_link, account["firefox_profile"], account["id"], account["name"], account["topic"])

                info("Generating pitch for affiliate marketing")
                afm.generate_pitch()
                info("Sharing pitch on Twitter")
                afm.share_pitch("twitter")
        else:
            info("Displaying cached products")
            table = PrettyTable()
            table.field_names = ["ID", "Affiliate Link", "Twitter Account UUID"]

            for product in cached_products:
                table.add_row([cached_products.index(product) + 1, colored(product["affiliate_link"], "cyan"), colored(product["twitter_uuid"], "blue")])

            print(table)

            user_input = question("Select a product to start: ")

            selected_product = None

            for product in cached_products:
                if str(cached_products.index(product) + 1) == user_input:
                    selected_product = product

            if selected_product is None:
                error("Invalid product selected. Restarting main function.")
                main()
            else:
                info(f"Selected product with affiliate link: {selected_product['affiliate_link']}")
                # Find the account
                account = None
                for acc in get_accounts("twitter"):
                    if acc["id"] == selected_product["twitter_uuid"]:
                        account = acc

                afm = AffiliateMarketing(selected_product["affiliate_link"], account["firefox_profile"], account["id"], account["name"], account["topic"])

                info("Generating pitch for affiliate marketing")
                afm.generate_pitch()
                info("Sharing pitch on Twitter")
                afm.share_pitch("twitter")

    elif user_input == 4:
        info("Starting Outreach...")

        outreach = Outreach()
        outreach.start()
    elif user_input == 5:
        if get_verbose():
            info(" => Quitting...", False)
        info("Exiting application")
        sys.exit(0)
    else:
        error("Invalid option selected. Restarting main function.")
        main()

if __name__ == "__main__":
    info("Starting application")
    print_banner()
    info("Printed ASCII banner")

    first_time = get_first_time_running()
    info(f"First time running: {first_time}")

    if first_time:
        info("First time setup initiated")
        print(colored("Ah! first time? No worries! Let's get you setup first!", "yellow"))

    info("Setting up file tree")
    assert_folder_structure()

    info("Removing temporary files")
    rem_temp_files()

    info("Fetching MP3 files")
    fetch_Music()

    # Start Flask server in background
    server = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 7860})
    server.daemon = True
    server.start()
    info("WebUI started at http://localhost:7860")

    while True:
        main()
