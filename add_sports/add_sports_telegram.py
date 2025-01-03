from itertools import islice

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (ApplicationBuilder, CallbackQueryHandler,
                          CommandHandler, ContextTypes)

from utils.constants import SPORTS_LOGO


class AddSportsTelegram:

    def __init__(self, user: str, password: str, host: str, port: str, database: str, table: str,
                 token_telegram_add_sport: str, chat_id_telegram: str, **kwargs):
        self.db_user = user
        self.db_password = password
        self.db_host = host
        self.db_port = port
        self.db_database = database
        self.db_table = table
        self.token = token_telegram_add_sport
        self.chat_id = chat_id_telegram
        self.engine = None
        self.session = None
        self.current_event_index = 0  # Track the current event index
        self.events_to_update = []  # Store events that need updating
        self._instantiate()

    def _instantiate(self) -> None:
        """Instantiate the database engine"""
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_database}"
        )
        self.session = sessionmaker(bind=self.engine)()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start command to show available events needing sports."""
        await self.load_events()  # Load events that need updating
        await self.check_and_ask_sports(update)  # Start checking for sports

    async def load_events(self):
        """Load events from the database that need a sport."""
        df = self.return_empty_titles()
        self.events_to_update = df.to_dict('records')  # Convert DataFrame to list of dicts
        self.current_event_index = 0  # Reset index

    def return_empty_titles(self) -> pd.DataFrame:
        """Return the data of all the events without a sport yet."""
        SELECT_QUERY = f"SELECT * FROM {self.db_table} WHERE sport IS NULL OR sport = ''"
        df = pd.read_sql(SELECT_QUERY, self.engine)
        return df

    async def check_and_ask_sports(self, update: Update):
        """Check for events without a sport and display them."""
        if self.current_event_index < len(self.events_to_update):
            event = self.events_to_update[self.current_event_index]
            title = event["title"]
            sub_title = event["sub_title"]
            self.current_id = event["ID"]
            await self.ask_user_for_sport(update, title, sub_title)
        else:
            if update.message:
                await update.message.reply_text("No more events to update. Use /start to check again.")
            elif update.callback_query:
                await update.callback_query.message.edit_text("No more events to update. Use /start to check again.")

    async def ask_user_for_sport(self, update, title, sub_title):
        """Display the title and sub_title of the event and sports menu."""
        message = f"title: {title}\nsub_title: {sub_title}"

        def chunked(iterable, size):
            """Split iterable into chunks of specified size."""
            it = iter(iterable)
            return iter(lambda: list(islice(it, size)), [])

        keyboard = [
            [InlineKeyboardButton(emoji, callback_data=sport) for emoji, sport in chunk]
            for chunk in chunked(SPORTS_LOGO.items(), 3)
        ]
        keyboard.append([InlineKeyboardButton("Other", callback_data="other")])
        keyboard.append([InlineKeyboardButton("Quit", callback_data="quit")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        if update.message:
            await update.message.reply_text(message, reply_markup=reply_markup)
        elif update.callback_query:
            await update.callback_query.message.edit_text(message, reply_markup=reply_markup)

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle sport selection."""
        query = update.callback_query
        await query.answer()

        sport = query.data
        if sport == "quit":
            await query.edit_message_text("Quitting the update process. Use /start to begin again.")
            self.pending_event = None  # Clear the current event
            return
        elif sport != "other":
            # Update the database with the selected sport
            update_query = text(f"UPDATE {self.db_table} SET sport = :sport WHERE id = :id")
            self.session.execute(update_query, {"sport": sport, "id": self.current_id})
            self.session.commit()

        # Move to the next event
        self.current_event_index += 1
        await self.check_and_ask_sports(update)  # Proceed to the next event

    def __call__(self):
        """Run the Telegram bot."""
        self.app = ApplicationBuilder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CallbackQueryHandler(self.button))

        print("Bot is running...")
        self.app.run_polling()