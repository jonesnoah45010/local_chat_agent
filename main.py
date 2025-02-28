from local_agent import ollama_chat_agent




if __name__ == "__main__":



	# #  Analyze a document that won't fit within the context window.
	# agent = ollama_chat_agent(name="Tomatio", model="llama3.2")

	# with open("docs/AliceInWonderland.txt", "r") as file:
	# 	book = file.read()

	# agent.semantic_db.purge_collection();
	# agent.semantic_db.insert_in_chunks(book, metadata={"doc_name":"Alice In Wonderland"}, max_sentences_per_chunk=5)

	# semantic_contextualize_prompt = "These are some passages from Alice in Wonderland"
	# semantic_query = "Interactions between Alice and the Mad Hatter"
	# agent.semantically_contextualize(semantic_query, semantic_top_k=5,
	#  semantic_where={"doc_name":"Alice In Wonderland"},
	#   semantic_contextualize_prompt=semantic_contextualize_prompt)

	# message = "Tell me about the relationship between Alice and the Mad Hatter. Use examples from the provided passages"
	# response = agent.chat(message)



	# have a generic chat
	agent = ollama_chat_agent(name="Tomatio", model="llama3.2")
	print("\n")
	while True:
		prompt = input("You: ")
		response = agent.chat(prompt,stream=True,show=True)
		print("\n")
		if prompt == "bye":
			break



	# # test out save and load ability
	# agent = ollama_chat_agent(name="Tomatio", model="llama2-uncensored")
	# response = agent.chat("say something very dirty you might say to a person with a big butt",stream=False)

	# agent.save_agent("Tomatio_copy.pkl")
	# agent2 = ollama_chat_agent.load_agent("Tomatio_copy.pkl")

	# ab = agent.save_agent(keep_in_memory=True)
	# agent3 = ollama_chat_agent.load_agent(data=ab,keep_in_memory=True)

















