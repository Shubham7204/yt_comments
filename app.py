import os
import re
import urllib.parse
import streamlit as st
import googleapiclient.discovery
import pandas as pd
from textblob import TextBlob
from langdetect import detect, LangDetectException
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize YouTube API
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = os.getenv("DEVELOPER_KEY")

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

# Define question words
question_words = ["how", "where", "what", "when", "?", "who"]

def fetch_comments(video_id, max_results=500):
    all_comments = []
    next_page_token = None
    while len(all_comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(all_comments)),
            pageToken=next_page_token
        )
        response = request.execute()

        comments = [
            {
                'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt'],
                'updated_at': item['snippet']['topLevelComment']['snippet']['updatedAt'],
                'like_count': item['snippet']['topLevelComment']['snippet']['likeCount'],
                'text': item['snippet']['topLevelComment']['snippet']['textDisplay']
            }
            for item in response['items']
        ]
        all_comments.extend(comments)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return all_comments[:max_results]  # Return only the requested number of comments

def analyze_sentiment(comment):
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def categorize_comments(comments):
    categorized_comments = {
        'Positive': [],
        'Negative': [],
        'Neutral': [],
        'Questions': []
    }
    for comment in comments:
        text = comment['text']
        sentiment = analyze_sentiment(text)
        
        if any(word.lower() in text.lower() for word in question_words):
            categorized_comments['Questions'].append(comment)
        else:
            categorized_comments[sentiment].append(comment)
    return categorized_comments

def create_knowledge_graph(hashtag, categorized_comments):
    graph = nx.DiGraph()
    positive_count = len(categorized_comments['Positive'])
    negative_count = len(categorized_comments['Negative'])
    neutral_count = len(categorized_comments['Neutral'])
    question_count = len(categorized_comments['Questions'])
    total_comments_count = positive_count + negative_count + neutral_count + question_count
    
    graph.add_node('Positive', count=positive_count)
    graph.add_node('Negative', count=negative_count)
    graph.add_node('Neutral', count=neutral_count)
    graph.add_node('Questions', count=question_count)
    graph.add_node('Total Comments', count=total_comments_count)
    graph.add_edge(hashtag, 'Total Comments', weight=total_comments_count)
    graph.add_edge(hashtag, 'Positive', weight=positive_count)
    graph.add_edge(hashtag, 'Negative', weight=negative_count)
    graph.add_edge(hashtag, 'Neutral', weight=neutral_count)
    graph.add_edge(hashtag, 'Questions', weight=question_count)

    pos = nx.spring_layout(graph, k=1.3)
    node_colors = ['lightgreen', 'lightcoral', 'lightblue', 'lightyellow', 'lightgray']
    for i, node in enumerate(['Positive', 'Negative', 'Neutral', 'Questions', 'Total Comments']):
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_color=node_colors[i], node_size=1500, alpha=0.8)

    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5, arrowsize=10)
    labels = {node: f"{node} ({graph.nodes[node].get('count', 0)})" for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight='bold')

    plt.title(f'Knowledge Graph for Hashtag: {hashtag}')
    filename = f'{hashtag}_knowledge_graph.png'
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def generate_pie_chart(categorized_comments, hashtag):
    sizes = [
        len(categorized_comments['Positive']),
        len(categorized_comments['Negative']),
        len(categorized_comments['Questions']),
        len(categorized_comments['Neutral'])
    ]
    labels = ['Positive', 'Negative', 'Questions', 'Neutral']
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0']
    explode = (0.1, 0, 0, 0)
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, shadow=True, wedgeprops={'edgecolor': 'gray'})
    plt.axis('equal')
    
    plt.title(f'Sentiment Distribution for Hashtag: {hashtag}')
    filename = f'{hashtag}_Pie_chart.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def generate_word_cloud(comments, hashtag):
    combined_comments = ' '.join(comment['text'] for comment in comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_comments)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Hashtag: {hashtag}')
    plt.axis('off')
    
    filename = f'{hashtag}_Word_Cloud.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def main():
    st.title("YouTube Comment Sentiment Analysis")
    
    video_url = st.text_input("Enter YouTube Video URL")
    
    if video_url:
        video_id = extract_video_id(video_url)
        if video_id:
            comments = fetch_comments(video_id)
            categorized_comments = categorize_comments(comments)
            
            # Display Comments Tables
            st.subheader("All Comments")
            st.write(pd.DataFrame(comments))
            
            st.subheader("Positive Comments")
            st.write(pd.DataFrame(categorized_comments['Positive']))
            
            st.subheader("Negative Comments")
            st.write(pd.DataFrame(categorized_comments['Negative']))
            
            st.subheader("Neutral Comments")
            st.write(pd.DataFrame(categorized_comments['Neutral']))
            
            st.subheader("Questions")
            st.write(pd.DataFrame(categorized_comments['Questions']))
            
            # Generate and display charts and knowledge graph
            knowledge_graph_image = create_knowledge_graph(video_id, categorized_comments)
            st.image(knowledge_graph_image)
            
            pie_chart_image = generate_pie_chart(categorized_comments, video_id)
            st.image(pie_chart_image)
            
            word_cloud_image = generate_word_cloud(comments, video_id)
            st.image(word_cloud_image)
        else:
            st.error("Invalid YouTube URL.")
    else:
        st.warning("Please enter a YouTube video URL.")

def extract_video_id(url):
    query = urllib.parse.urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urllib.parse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    raise ValueError('Invalid YouTube URL or unable to extract video ID.')

if __name__ == "__main__":
    main()