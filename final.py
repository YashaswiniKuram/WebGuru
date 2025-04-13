import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
from urllib.parse import urljoin, urlparse
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

genai.configure(api_key='AIzaSyC1yN5KGES_TZgsK_ezQER7gL244vON2f8')

qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
#qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
#qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

website_content = []
content_embeddings = []

def scrape_website(base_url: str, max_depth: int = 1):
    """Iteratively scrape a website to collect content and links up to a specified depth.Max Depth is 1 by default.Change it using max_depth parameter."""
    visited_pages = set()
    queue = [(base_url, 0)]  # Queue stores tuples of (URL, current depth)
    
    while queue:
        url, current_depth = queue.pop(0)  
        if url in visited_pages or current_depth > max_depth:
            continue
        
        visited_pages.add(url)
        
        try:
            response = requests.get(url)
            response.raise_for_status()  
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.RequestException:
            continue  
        page_text = soup.get_text(separator=" ").strip()
        website_content.append({"url": url, "content": page_text})
        embedding = embedding_model.encode(page_text)  
        content_embeddings.append(embedding)
        for link in soup.find_all("a", href=True):
            link_url = urljoin(base_url, link['href'])
            if link_url.startswith(base_url) and link_url not in visited_pages:
                queue.append((link_url, current_depth + 1))

def get_all_links(url):
    """Get all links from a single page."""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = set()  
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            full_link = urljoin(url, link)  
            if urlparse(full_link).scheme in ['http', 'https']:
                links.add(full_link)
        
        return links
    except Exception as e:
        print(f"Error fetching links: {e}")
        return set()

def visualize_links(main_url, links, output_image_path="website_structure.png"):
    """Visualize the structure of the website as a graph."""
    G = nx.DiGraph()
    G.add_node(main_url)
    for link in links:
        G.add_node(link)
        G.add_edge(main_url, link)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)  
    plt.figure(figsize=(12, 12))  
    
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold', arrows=True)
    
    plt.title(f"Website Structure: {main_url}")
    plt.savefig(output_image_path, format="PNG")
    plt.close()
    print(f"Visualization saved as {output_image_path}")

def generate_summary(transcript_text):
    """Summarize content using Google Gemini model."""
    structured_prompt = ("""You are a website summarizer. Provide a summary of the following content within 300-500 words:\n\n""" f"Here is the content:\n\n{transcript_text}")
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(structured_prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"

def answer_question(question):
    """Answer questions based on scraped content using the QA model."""
    question_embedding = embedding_model.encode(question)
    similarities = util.cos_sim(question_embedding, content_embeddings)[0]
    closest_idx = int(similarities.argmax())  
    
    content = website_content[closest_idx]["content"]
    url = website_content[closest_idx]["url"]
    answer = qa_pipeline(question=question, context=content)
    
    return answer["answer"], answer["score"], url

def save_summary_to_file(summary, filename="static/website_summary.txt"):
    """Save the summary to a text file for download."""
    try:
        with open(filename, "w") as file:
            file.write(summary)
        return filename
    except Exception as e:
        return None

def main():
    start_url = input("Enter the URL of the website to scan: ")
    
    print("Scraping website content...")
    scrape_website(start_url)
    print("Website content scraped successfully.")
    
    print("Fetching all links for structure visualization...")
    links = get_all_links(start_url)
    visualize_links(start_url, links)
    
    print("\nGenerating summaries for main page...")
    for page in website_content:
        if page['url'] == start_url:
            summary = generate_summary(page["content"])
            print(f"\nSummary for {page['url']}:\n{summary}")
    
    print("\nReady to answer questions.")
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        answer, score, url = answer_question(question)
        print(f"Answer: {answer} (from: {url})\nConfidence Score: {score}\n")

if __name__ == "__main__":
    main()
