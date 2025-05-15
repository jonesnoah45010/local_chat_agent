from local_agent import ollama_chat_agent




if __name__ == "__main__":



	#  Analyze a document without having to upload entire document into context window
	agent = ollama_chat_agent(name="Tomatio", model="llama3.2")

	agent.semantic_db.purge_collection();

	agent.upload_document("docs/AliceInWonderland.txt")

	semantic_query = "Interactions between Alice and the Mad Hatter"
	agent.discuss_document(semantic_query, doc_name="AliceInWonderland.txt")

	message = "Tell me about the relationship between Alice and the Mad Hatter. Use examples from the provided passages"
	response_stream = agent.chat(message)

	agent.print_stream(response_stream)







	# # have a generic chat
	# agent = ollama_chat_agent(name="Bob", model="llama3.2")
	# while True:
	# 	prompt = input("You: ")
	# 	response_stream = agent.chat(prompt)
	# 	agent.print_stream(response_stream)
	# 	if prompt == "bye":
	# 		break







	# # test out save and load ability
	# agent = ollama_chat_agent(name="Tomatio", model="llama2-uncensored")
	# response = agent.chat("say something unsulting and gross",stream=False)

	# agent.save_agent("Tomatio_copy.pkl")
	# agent2 = ollama_chat_agent.load_agent("Tomatio_copy.pkl")

	# ab = agent.save_agent(keep_in_memory=True)
	# agent3 = ollama_chat_agent.load_agent(data=ab,keep_in_memory=True)




