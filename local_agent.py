import requests
import json
from local_agent_tools import *
import tiktoken
from local_semantic_db import local_semantic_db
from local_sql_db import local_sql_db
from basic_web_scraping import basic_web_scrape, contains_url, extract_urls
import pickle





class ollama_chat_agent:
	def __init__(self, model="llama3.2", model_encoding=None, name="Agent", url="http://localhost:11434/api", context_window_limit=2048):
		"""
			concversation_history: list of dictionaries like... 
				[{"role":"user","content":"hello"}, 
				{"role":"assistant","content":"hi"},
				{"role": "system", "content": "you are an assistant"}]
				... representing the current conversation.
			model: name of the local Ollama model you are using
				and have installed on your machine.
			enc: encoding used by your local model for counting tokens
			url: where local HTTP requests can be sent to Ollama api
			name: name you want the agent to use as their name.
			context_window_limit: how many tokens the model supports for 
				its context window, i.e. how long the current conversation
				can be before it needs to be refreshed or end.
		"""
		self.conversation_history = []
		self.model=model
		self.model_encoding=model_encoding
		self.enc=self.figure_out_model_encoding()
		self.url=url
		self.name=name
		self.context_window_limit=context_window_limit
		if type(name) is str:
			self.add_context("Your name is " + self.name)
		# self.semantic_db = local_semantic_db(persist_directory=str(self.name+"_data"+"/"+self.name+"_semantic_db"), collection_name=str(self.name+"_general"))
		# self.sql_db = local_sql_db(str(self.name+"_data"+"/"+self.name+"_sql_db"))

		self._initialize_databases()

	def _initialize_databases(self):
		"""
		Initializes the semantic and SQL databases.
		This method is called upon instantiation and after unpickling.
		"""
		self.semantic_db = local_semantic_db(
			persist_directory=f"{self.name}_data/{self.name}_semantic_db", 
			collection_name=f"{self.name}_general"
		)
		self.sql_db = local_sql_db(f"{self.name}_data/{self.name}_sql_db")


	def figure_out_model_encoding(self):
		if type(self.model_encoding) is str:
			try:
				return tiktoken.encoding_for_model(self.model_encoding)
			except:
				return tiktoken.encoding_for_model("gpt-3.5-turbo")
		else:
			try:
				return tiktoken.encoding_for_model(self.model)
			except:
				return tiktoken.encoding_for_model("gpt-3.5-turbo")

	def token_count(self, message):
		"""
			returns how many tokens a given message will be counted as for the currently used model
		"""
		return len(self.enc.encode(str(message)))

	def current_tokens_used(self):
		"""
			returns how many tokens are currently being used for the entire conversation history
		"""
		return self.token_count(self.conversation_history)

	def is_within_context_window(self, context_window_limit=None, current_message=None):
		"""
			returns True if the current conversation history is within the context_window_limit
		"""
		if context_window_limit is None:
			context_window_limit = self.context_window_limit
		current_token_count = self.current_tokens_used()
		if current_message is not None:	
			current_token_count += self.token_count({"role": "user", "content": current_message})
		return current_token_count <= context_window_limit

	def tokens_left(self):
		"""
			returns how many tokes are left before the current conversation goes out of the context_window_limit
		"""
		context_window_limit = self.context_window_limit
		t = self.current_tokens_used()
		return int(context_window_limit) - int(t)

	def refresh_conversation(self, summary_size=500, summarize_prompt=None, summary_reference_prompt=None, show=False):
		"""
			summarizes the current conversation, deletes the conversation history and then inserts the 
			summary as the new conversation history.  Used to avoid going out od the context_window_limit
		"""
		if summarize_prompt is None:
			summarize_prompt = "Summarize the following conversation ... "
		summarize_prompt += str(self.conversation_history)
		summary = self.respond(message=summarize_prompt,show=show)
		if summary_reference_prompt is None:
			summary_reference_prompt = "This is a summary of the conversation so far ... " + str(summary)
		else:
			summary_reference_prompt += " " + str(summary)
		self.conversation_history = []
		self.add_context(summary_reference_prompt)

	def add_context(self, system_prompt):
		"""
			add system prompts to contextualize the current conversation
		"""
		self.conversation_history.append({"role": "system", "content": system_prompt})

	def simple_generate_request(self, message):
		"""
		Sends a simple HTTP request to Ollama API and returns the raw response text.
		"""
		payload = {
			"model": self.model,
			"prompt": message,
			"stream": False  # Ensures the response is returned as a whole
		}
		headers = {"Content-Type": "application/json"}

		response = requests.post(self.url+"/generate", headers=headers, json=payload)

		if response.status_code == 200:
			data = response.json()
			return data.get("response", "").strip()
		else:
			return f"Error: {response.status_code}, {response.text}"


	def respond(self, message, show=True, finish_stream=True, stream=True):
		"""
			sends HTTP POST request with message to Ollama API and returns response
		"""
		payload = {
			"model": self.model,
			"prompt": message,
		}
		headers = {"Content-Type": "application/json"}
		response = requests.post(self.url+"/generate", headers=headers, data=json.dumps(payload), stream=stream)
		if finish_stream is False:
			return response
		full_text = ""
		if show:
			print(str(self.name)+": ")
		for chunk in response.iter_lines(decode_unicode=True):
			if chunk:
				data = json.loads(chunk)
				word = data.get("response", "").strip().replace("\n","")
				if word:
					if show:
						print(word, end=" ", flush=True)
					full_text += word + " "
		return full_text


	def semantically_contextualize(self, message, semantic_top_k=1, semantic_where=None, semantic_contextualize_prompt=None):

		context = self.semantic_db.query(message, top_k=semantic_top_k, where=semantic_where)
		context_text = []
		for d in context:
			context_text.append(d["text"])
		context = " ... ".join(context_text)

		# Include semantic context if applicable
		if context is not None:
			print("semantically_contextualize: context added")
			if semantic_contextualize_prompt is None:
				self.add_context("This information may be relevant to the conversation ... " + str(context))
			else:
				self.add_context(str(semantic_contextualize_prompt) + " ... " + str(context))
		else:
			print("semantically_contextualize: No context added")



	def chat(self, message, show=True, stream=True, conversation=True, auto_refresh=True, 
			 show_tokens_left=False, refresh_summary_size=500, refresh_summarize_prompt=None, 
			 refresh_summary_reference_prompt=None):

		"""
		Sends a message to the agent.
		If show is True, then logging messages and the conversation stream will be printed to the 
		console. 
		If conversation is True, then user and agent messages are logged to the conversation_history. 
		If auto_refresh is True, then the conversationhistory will be summarized and reset to this 
		summary once the conversation has reached a point where it is about to go out of the 
		context window limit. 
			When auto_refresh is enabled the refresh_summary_size determines how large the summary of 
			the conversation will be once the refresh occurs.
			refresh_summarize_prompt is optional and can be used as the prompt to be used to execute
			the summarization.  If left as None, will default to "Summarize the following conversation ... ".
			refresh_summary_reference_prompt will be the prompt prepended to the conversation_history
			after the summarization occurs.  If left as None, will default to 
			"This is a summary of the conversation so far ... "
		If check_semantic_db is True, then the current message will be contextualized using information found
		in the semantic database.
			When check_semantic_db is enabled semantic_top_k will determine how many entries are potentially
			pulled from the semantic database.  i.e. semantic_top_k=3 means up to 3 results could be pulled.
			semantic_top_k will default to 1.
			semantic_where can be set to a condition like {name="bob"} that will be used to filter the 
			semantic query to the semantic database.  If left as None, no filter will be used.
			semantic_contextualize_prompt is prepended to the context before it is inserted into the
			conversation or appended to the message. If left as None, will default to 
			"This information may be relevent to the conversation ... "

		"""

		# Append user message to conversation history
		if conversation:
			self.conversation_history.append({"role": "user", "content": message})

		if show_tokens_left and auto_refresh:
			print(f"{self.name}: Tokens left until refresh: {self.tokens_left()}")

		# Check if context window is about to be exceeded
		if auto_refresh and not self.is_within_context_window(current_message=message):
			if show:
				print(f"{self.name}: CONTEXT WINDOW LIMIT ABOUT TO GO OUT OF BOUNDS, REFRESHING CONVERSATION")
			self.refresh_conversation(summary_size=refresh_summary_size, 
									  summarize_prompt=refresh_summarize_prompt, 
									  summary_reference_prompt=refresh_summary_reference_prompt)


		# Properly format the conversation history for Ollama API
		payload = {
			"model": self.model,
			"messages": self.conversation_history  # Pass full conversation history
		}

		headers = {"Content-Type": "application/json"}

		# Send request to Ollama API
		response = requests.post(self.url+"/chat", headers=headers, json=payload, stream=stream)

		# return response
		full_text = ""

		if show:
			print(self.name +": ", end="")
		if response.status_code == 200:
			if stream:
				# Process each line separately (NDJSON format)
				for chunk in response.iter_lines(decode_unicode=True):
					# print("STREAM: " + str(chunk))
					if chunk:
						try:
							data = json.loads(chunk)  # Parse each JSON object individually
							content = data.get("message", {}).get("content", "")
							if content:
								if show:
									print(content, end="")
								full_text += content

						except json.JSONDecodeError:
							continue  # Skip malformed JSON lines
			else:
				# Handle non-streamed response (which is still NDJSON format)
				try:
					json_objects = response.text.strip().split("\n")  # Split NDJSON into lines
					for obj in json_objects:
						data = json.loads(obj)  # Parse each JSON object
						content = data.get("message", {}).get("content", "")
						# print("NO STREAM: " + str(content))
						if content:
							full_text += content  # Concatenate response parts
				except json.JSONDecodeError:
					print("Error: Could not decode JSON response")
					full_text = f"Error: Invalid JSON response from Ollama API"
		else:
			full_text = f"Error: {response.status_code}, {response.text}"
		
		# Append assistant response to conversation history
		if conversation and full_text != "":
			self.conversation_history.append({"role": "assistant", "content": full_text})


		return full_text



	def save_agent(self, filepath=None, keep_in_memory=False):
		"""
		Saves the current instance of ollama_chat_agent to a pickle file or returns raw bytes if keep_in_memory is True.
		Excludes semantic_db and sql_db to avoid potential pickling issues.
		"""
		try:
			data_to_save = self.__dict__.copy()  # Create a copy of the object's attributes
			data_to_save.pop("semantic_db", None)  # Remove database attributes before pickling
			data_to_save.pop("sql_db", None)
			
			pickled_data = pickle.dumps(data_to_save)
			
			if keep_in_memory:
				return pickled_data
			
			if filepath:
				with open(filepath, 'wb') as f:
					f.write(pickled_data)
				print(f"Agent saved successfully to {filepath}")
			
		except Exception as e:
			print(f"Error saving agent: {e}")
			return None

	@staticmethod
	def load_agent(data, keep_in_memory=False):
		"""
		Loads an instance of ollama_chat_agent from a pickle file or from raw bytes if keep_in_memory is True.
		Reinitializes semantic_db and sql_db after loading.
		"""
		try:
			if keep_in_memory:
				saved_data = pickle.loads(data)
			else:
				with open(data, 'rb') as f:
					saved_data = pickle.load(f)
			
			# Create a new instance and restore its attributes
			agent = ollama_chat_agent()
			agent.__dict__.update(saved_data)
			
			print("Agent loaded successfully")
			return agent
		except Exception as e:
			print(f"Error loading agent: {e}")
			return None













if __name__ == "__main__":
	agent = ollama_chat_agent(name="Bob", model="llama3.2")
	while True:
		prompt = input("You:\n")
		response = agent.chat(prompt)
		print("\n")
		if prompt == "bye":
			break

























