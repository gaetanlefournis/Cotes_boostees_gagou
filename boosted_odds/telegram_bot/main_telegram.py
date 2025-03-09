from telegram import Bot


class TelegramBot():
    """Class to interact with the telegram bot
    """
    def __init__(self, token : str, chat_id : str) -> None:
        self.token = token
        self.chat_id = chat_id
        self.bot = None
        self._initiate()

    def _initiate(self) -> None:
        """Initiate the telegram bot"""
        self.bot = Bot(token=self.token)

    async def send_boosted_odd_to_telegram(self, data : dict) -> None:
        """Send the boosted odd to telegram"""
        try:
            message = f"site : {data['website']}, \nsport: {data['sport']}, \ntitle : {data['title']}, \nsubtitle : {data['sub_title']}, \nold_odd: {data['old_odd']}, \nnew_odd: {data['odd']}"
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"An error occurred while sending boosted odd: {e}")
