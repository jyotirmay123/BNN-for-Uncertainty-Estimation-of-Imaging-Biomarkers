import os
import torch
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import json

class CommonUtils(object):
    def __init__(self):
        super().__init__()
        self.elem1 = None

    @staticmethod
    def create_if_not(path):
        """
        Creates a folder at the given path if one doesnt exist before
        ===

        :param path: destination to check for existense
        :return: None
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def create_file_if_not(path):
        """
        Creates a folder at the given path if one doesnt exist before
        ===

        :param path: destination to check for existense
        :return: None
        """
        dir = '/'.join(path.split('/')[:-1])
        if not os.path.exists(dir):
            os.makedirs(dir)
            file = open(path, 'w')
            file.close()

    @staticmethod
    def reparameterize(mu, logvar):
        """

        :param mu: mean
        :param logvar: variance
        :return: randomly generated sample
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def setup_whatsapp_notifier(self, driver_path='/home/abhijit/Jyotirmay/thesis/hquicknat/utils/chromedriver', receiver="Tum Madhu Mathematics"):
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