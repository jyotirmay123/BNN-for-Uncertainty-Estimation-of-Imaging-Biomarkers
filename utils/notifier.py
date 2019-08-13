
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.common.by import By
import json

import smtplib

from string import Template

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class Notifier(object):
    def __int__(self):
        super().__init__()
        self.elem1 = None
        self.s = None
        self.MY_ADDRESS = None

    def setup_whatsapp_notifier(self, driver_path='/home/abhijit/Jyotirmay/thesis/hquicknat/utils/chromedriver',
                                receiver="Tum Madhu Mathematics"):
        driver = webdriver.Chrome(driver_path)
        driver.get('https://web.whatsapp.com')
        input('click "enter" once whatsapp configured in selenium invoked browser!')
        print("Success!!!")
        spans = driver.find_elements_by_tag_name('span')
        elem_lst = [x for x in spans if x.get_attribute('title') == 'Tum Madhu Mathematics']
        elem = elem_lst[0]
        elem.click()
        self.elem1 = driver.find_element_by_class_name('_3u328')

    def whatsapp_notifier(self, message):
        self.elem1.send_keys(json.dumps(message) + '\r', Keys.RETURN)
        print('Notified in whatsapp')

    def setup_mail_notifier(self ):
        self.MY_ADDRESS = 'senapati.jyotirmay@gmail.com'
        PASSWORD = 'Shaan@lostworld1234.'

        # set up the SMTP server
        self.s = smtplib.SMTP(host='smtp.gmail.com', port=587)
        self.s.starttls()
        self.s.login(self.MY_ADDRESS, PASSWORD)

    def mail_notifier(self, message):
        msg = MIMEMultipart()  # create a message

        # setup the parameters of the message
        msg['From'] = self.MY_ADDRESS
        msg['To'] = 'j.senapati@tum.de'
        msg['Subject'] = "project_notification"

        # add in the message body
        msg.attach(MIMEText(json.dumps(message), 'plain'))

        # send the message via the server set up earlier.
        self.s.send_message(msg)
        del msg

    def __del__(self):
        if self.s is not None:
            self.s.quit()
        if self.elem1 is not None:
            del self.elem1
        print('Destructor called, Notifier deleted.')