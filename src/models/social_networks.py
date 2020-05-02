import codecs
import json
import re
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from nltk.util import ngrams
from segtok.tokenizer import word_tokenizer
from tqdm import tqdm

from src.preprocessing import steps


class SocialNetwork:
    # TODO: define additional preprocessing step with pass (?)
    def __init__(self, path, name, n_gram, ignore, **kwargs):
        self.path = path
        self.n_gram = n_gram
        self.name = name
        self.ignore = ignore
        self.kwargs = kwargs
        self.lm_dict = dict()
        self.messages_list = []

    @abstractmethod
    def _get_messages(self):
        pass

    def _n_grams(self):
        """Make n_grams dict from list of messages."""
        main_dict = defaultdict(dict)
        for record in self.messages_list:
            tokenized = word_tokenizer(record)
            # TODO: find a better way to insert start/end tokens
            n_gramed = ngrams(
                tokenized,
                self.n_gram,
                pad_left=True,
                pad_right=True,
                left_pad_symbol="<start_token>",
                right_pad_symbol="<end_token>",
            )
            for n_gram in n_gramed:
                # instead of counter, so we iterate over only once
                # but with if statement
                # TODO: estimate speed of two approaches
                if main_dict.get(n_gram[:-1], {}).get(n_gram[-1]):
                    main_dict[n_gram[:-1]][n_gram[-1]] += 1
                else:
                    main_dict[n_gram[:-1]][n_gram[-1]] = 1
        self.lm_dict = main_dict

    def _preprocessing(self):
        self.messages_list = [steps.replace_url(x) for x in self.messages_list]
        self.messages_list = [steps.replace_html_tags(x) for x in self.messages_list]
        self.messages_list = [steps.replace_punct(x) for x in self.messages_list]
        self.messages_list = steps.remove_non_russian(self.messages_list)

    def process(self):
        self._get_messages()
        self._preprocessing()
        self._n_grams()
        return self

    def save(self, path, path_list):
        with open(path, "w") as file_:
            json.dump(self.lm_dict, file_, ensure_ascii=False, indent=4)
        with open(path_list, "w") as file_:
            json.dump(self.messages_list, file_, ensure_ascii=False, indent=4)


class Telegram(SocialNetwork):
    def _get_messages(self):
        person = self._get_messages_by_name()
        self.messages_list = [self._ensure_str(x) for x in person]

    def _get_messages_by_name(self):
        chats = self._data_loader()
        list_of_messages = []

        for chat in chats:
            if chat["name"] not in self.ignore:
                for message in chat["messages"]:
                    if not message.get("action"):  # remove calls
                        if message["from"] == self.name:  # remove others
                            if not message.get(
                                "forwarded_from"
                            ):  # remove forwarded messages
                                if message.get(
                                    "sticker_emoji"
                                ):  # get emoji if it's a sticker
                                    list_of_messages.append(message["sticker_emoji"])
                                if message.get("text"):  # finally get text
                                    list_of_messages.append(message["text"])
        return list_of_messages

    def _ensure_str(self, record):
        """Convert list-like message to string."""
        if isinstance(record, str):
            return record
        else:
            return " ".join(
                [x for x in record if isinstance(x, str) and x not in [" ", ""]]
            )

    def _data_loader(self):
        with open(self.path) as file_:
            telegram = json.load(file_)
        return telegram["chats"]["list"][1:]


# TODO: add removing @user
class VK(SocialNetwork):
    def _find_year(self, str_):
        regexp = re.compile("\d\d\d\d")
        return regexp.findall(str_)[0]

    def _get_chats_path(self):
        menu_html = Path(self.path)
        with menu_html.open() as file_:
            menu = BeautifulSoup(file_.read(), features="html.parser")
        vk_chats = [
            x.attrs["href"]
            for x in menu.find_all("a")
            if x.text not in self.ignore and x.text != "Профиль"  # remove ../index.html
        ]
        return [(menu_html.parent / x).parent for x in vk_chats]

    def _get_messages_from_page(self, filename):
        with codecs.open(filename, encoding="Cp1251") as file_:
            page = BeautifulSoup(file_.read(), features="html.parser")
        return page.find_all(attrs={"class": "message"})

    def _process_message(self, message):
        min_year = self.kwargs.get("min_year")
        attachment = message.find(attrs={"class": "attachment__description"})
        if attachment and attachment.text == "Сообщение удалено":
            return ""
        # TODO: replace `Вы` as parameter
        if message.find(attrs={"class": "message__header"}).text.startswith("Вы,"):
            if min_year and (
                int(
                    self._find_year(
                        message.find(attrs={"class": "message__header"}).text
                    )
                )
                > min_year
            ):
                message.find(attrs={"class": "message__header"}).decompose()
                for kludge in message.find_all(attrs={"class": "kludges"}):
                    kludge.decompose()
                if message.br:
                    message.br.replace_with(" ")
                return message.text.replace("\n", " ").strip()
            else:
                return ""

    def _get_texts_chat(self, path):
        texts = []
        for filename in path.iterdir():
            messages = self._get_messages_from_page(filename)
            for message in messages:
                texts.append(self._process_message(message))
        return [x for x in texts if x]

    def _get_messages(self):
        dataset = []
        chats = self._get_chats_path()
        for chat in tqdm(chats):
            chat_texts = self._get_texts_chat(chat)
            dataset.extend(chat_texts)
        self.messages_list = dataset
