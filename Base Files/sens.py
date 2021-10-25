#!/bin/python3

'''
TITLE: SDR EmComm Notification System (SENS)
BY: Common Sense Cyber Group

Developers:
    -Some Guy They Call Scooter
    -Common Sense Cyber Group

Version: 2.0.1

License: Open GNU 3

Created Date: 10/25/2021
Updated Date: 10/25/2021

Purpose:
    -This script is intended to be used as a real-time speech recognician and notification system for EmComm SDR scanners.
    -This script uses the Mozilla ASR engine as well as deepspeech in order to process speech from audio files and transcribe them to text.
    -This file can br run as a standalone script, as a CRON job, or as its intended purpose, alongside TrunkRecorder after a recording is saved (See README for official instructions)
    -This script will search each transmission for specific words, and if heard, a notification will be sent to the configured destinations for alerting
    -All speech recognition is done offline and logged (So in a case where there is no internet access, we can still decode what is being said in radio transmissions and it is logged)
    -The email portion (notification) of this script does require internet access to email the list of emails in the config file and let them know what trigger word was heard, as well as send them an attachment

Use Case:
    -The indended use of this is for emergency notification based on words used in police/fire/ems radio transmissions
    -This would be useful in an event that a specific area (say someones home, street, or a major event) is being discussed so a notification would be sent out alerting the user/s of the incident
    -This script and its functionality is meant to be as lightweight as possible so it can be run on a Raspberry Pi (3b+ or higher) to make portable, small, and low-power

Notes:
    -This script requires the setup of DeepSpeech as well as the VAD CMD tools from Mozilla ASR. See instructions here: https://github.com/mozilla/DeepSpeech-examples/tree/r0.9/vad_transcriber
    -If we run into issues with recordings being cut off if they are too long or have spaces, we may need to look at using this instead: https://github.com/mozilla/DeepSpeech-examples/tree/r0.9/vad_transcriber
    -This file will be run via a shell script that sits in the trunk-build dir each time a recording completes. See trunk-recorder documentation on how to configure this feature
    -If for some reason Numpy is giving you issues, try running 'sudo apt-get install libatlas-base-dev' (Debian)

To Do / Changes:
    -Set up attaching files to email notification so user is able to verify what waws said
    -Testing
    -If results are not accurate, may have to look at manual training, or cleaning up of the audio file

'''

### IMPORT LIOBRARIES ###
import sys              #https://docs.python.org/3/library/sys.html - Used for error catching
import os           #https://docs.python.org/3/library/os.html - Used for various OS activities with files
import logging          #https://docs.python.org/3/library/logging.html - Used for logging features of script
import smtplib          #https://docs.python.org/3/library/smtplib.html - Used for sending out email alerts to designated people if an alert word is triggered
import ssl              #https://docs.python.org/3/library/ssl.html - Used for sending out email alerts to designated people if an alert word is triggered
from datetime import datetime   #https://docs.python.org/3/library/datetime.html - Used for getting the current time that the trigger word was heard/alert sent
import subprocess               #https://docs.python.org/3/library/subprocess.html - Used to subprocess multiple instances of the sens.py script for faster alerting
import time         #https://docs.python.org/3/library/time.html - Used for waiting on processes to finish
from deepspeech import Model    #https://deepspeech.readthedocs.io/en/v0.9.3/ - Used for ASR processing of speech
import wave         #https://docs.python.org/3/library/wave.html - Used for opening and reading WAV audio files
import numpy as np  #https://numpy.org/ - Math things


### DEFINE VARIABLES ###
recordings_path = '/home/pi/SENS_Master/trunk_recorder/trunk-build/new_recordings/'
completed_path = '/home/pi/SENS_Master/trunk_recorder/trunk-build/processed_recordings/'
process_list = []  #List to hold the names of files that prossesing has been started on
model_path = '/home/pi/Deepspeech/deepspeech-0.9.3-models.tflite'
scorer_path = '/home/pi/Deepspeech/deepspeech-0.9.3-models.scorer'

#Set up logging
logging_file = 'SENS.log'         #Define log file location for windows
logger = logging.getLogger('SENS_Logging')  #Define log name
logger.setLevel(logging.DEBUG)              #Set logger level
fh = logging.FileHandler(logging_file)      #Set the file handler for the logger
fh.setLevel(logging.DEBUG)                  #Set the file handler log level
logger.addHandler(fh)                       #Add the file handler to logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')   #Format how the log messages will look
fh.setFormatter(formatter)                  #Add the format to the file handler


### FUNCTIONS ###
#Function to read in configuration file in order to get important and relevant values
def read_config():
    #Set global variables
    global detection_word_list, alert_list, use_alerting, alert_email, alert_password, smtp_port, smtp_server

    detection_word_list = []    #List of words that will set off alerts
    use_alerting = False        #Determines if we will use the alerting feature or not
    alert_list = []             #List of emails to send alerts to in the case of a detection

    #Open the config file
    try:
        with open('sens.conf') as file:
            rows = file.readlines()

            for row in rows:
                #Pull out all trigger words
                try:
                    if "trigger_word_" in row:
                        detection_word_list.append(row.split(":")[1].lower().replace("\n", ""))
                except:
                        logger.error("Unable to read trigger_words from config file! Please check syntax!")
                        quit()

                #Pull out if the user wants to set up alerting
                if "use_alerting" in row:
                    try:
                        if row.split(":")[1] == "True":
                            use_alerting = True
                    except:
                        logger.error("Unable to read use_alerting from config file! Please check syntax!")
                        quit()

                #Pull out the list of people to alert
                if "alert_user_" in row:
                    try:
                        if "@" not in row.split(":")[1] or "." not in row.split(":")[1]:
                            logger.error("Incorrectly formatted email in alert_user! Please check syntax!")

                        alert_list.append(row.split(":")[1])
                    except:
                        logger.error("Unable to read alert_users from config file! Please check syntax!")
                        quit()

                #Pull out the alert email and password
                if "alert_email" in row:
                    try:
                        if "@" not in row.split(":")[1] or "." not in row.split(":")[1]:
                            logger.error("Incorrectly formatted email in alert_user! Please check syntax!")

                        alert_email = row.split(":")[1]
                    except:
                        logger.error("Unable to read alert_email from config file! Please check syntax!")
                        quit()

                if "alert_passwd" in row:
                    try:
                        alert_password = row.split(":")[1]
                    except:
                        logger.error("Unable to read alert_password from config file! Please check syntax!")
                        quit()

                #Get the SMTP information
                if "smtp_server" in row:
                    try:
                        smtp_server = row.split(":")[1]
                        smtp_port = row.split(":")[2]
                    except:
                        logger.error("Unable to read SMTP info from config file! Please check syntax!")
                        quit()

    except:
        logger.critical("Error Occurred when opening config file! Closing!")
        quit()

#Function to process the results of the translations and send an alert if we catch a trigger word
def process_results(result):
    #Iterate through our list of trigger words and if one is found in the result, call the alert function
    for trigger in detection_word_list:
        if trigger in str(result).lower():
            logger.info("Trigger word was found in broadcast!  -  %s", trigger)

            #Call the send alert function to notify people of the trigger. Include the interpretation, the trigger word, and the file so we can send it in the alert
            #Break after we send the alert because we dont need to be spamming alerts for every trigger word in a sentence when we send the user the whole transcription
            send_alert(result, trigger, to_process)
            break

#Function to send out email alerts in the case that a detection word is found
def send_alert(full_message, trigger_word, audio_file):
    print("The detected Trigger word was: '", trigger_word, "' while processing.")

    #Create message to send
    message_beginning = """\
        SENS Alert! - Trigger word was detected on scanner!


        """

    message = f'{message_beginning}Trigger word: {trigger_word} was detected at {datetime.now()}\nFull translation is in {audio_file} and printed below!\n\n\t{full_message}'

    #Create a secure SSL context
    context = ssl.create_default_context()

    #Try to log in to server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(alert_email, alert_password)

        #Iterate through the list of contacts from the config file and send them the email
        for email in alert_list:
            try:
                server.sendmail(alert_email, email, message)
                logger.info("Sent notification to: %s", email)
            except:
                logger.error("Error sending alert email!!!")

    return

### THE THING ###
if __name__ == '__main__':
    try:
        #Read in config file information
        read_config()

        #Search for the new audio files and convert using ffmpeg
        try:
            for root, directories, file in os.walk(recordings_path):
                for item in file:
                    if(item.endswith(".mp4")) and item not in process_list:
                        process_list.append(item)
                        to_run = str(root) + "/" + str(item)

                        #Make name of file that we will actually process
                        to_process = "New_Recordings" + str(item) + ".wav"

                        #Convert to proper WAV format
                        p1 = subprocess.Popen("ffmpeg -i " + to_run + "-ar 16000 -ac 1 " + to_process)
                        
                        #Wait while the file is converted to a WAV
                        while p1.poll() is None:
                            time.sleep(5)

                        logger.info("Converted %s to WAV file for processing", to_run)

                        #Use ASR to transcribe the audio file offline
                        ds = Model(model_path)
                        ds.enableExternalScorer(scorer_path)
                        fin = wave.open(to_process, 'rb')
                        frames = fin.readframes(fin.getnframes())
                        audio = np.frombuffer(frames, np.int16)
                        transcription = ds.stt(audio)
                        logger.info("Transcription finished and heard: %s", transcription)

                        #Process the transcription to see if any trigger words were heard
                        process_results(transcription)

                        #Now that processing is done and has been logged, move the audio file to the completed folder so we don't process it again
                        remove_json = ""
                        remove_wav = ""
                        os.remove(to_run.replace(".mp4", ".wav")) #original WAV file generated by trunk-recorder that we do not use
                        os.remove(to_run.replace(".mp4", ".json"))  #json info file that goes with the recording
                        os.remove(to_run)   #original mp4 file that we don't really need anymore

                        completed_file = completed_path + to_process

                        os.rename(to_process, completed_file)

                        logger.info("Finished deleteing junk files and moving original file to completed folder: %s", to_process)

        except KeyboardInterrupt:
            logger.info("User closed script with Ctrl-C")
            quit()
        except:
            logger.critical("Unexpected error: %s", sys.exc_info())

    except:
        logger.critical("Unexpected error: %s", sys.exc_info())