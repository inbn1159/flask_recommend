from scipy.sparse import load_npz
import pandas as pd
import pymysql
from sklearn.metrics.pairwise import cosine_similarity
from kiwipiepy import Kiwi
import urllib.parse
from flask import Flask, request, jsonify
import re
import pickle
import time

# DB 접속용
config = {
    'user': 'test-user',
    'password': '1234qwer',
    'host': '49.50.167.140',
    'port': 3306,
    'database': 'testdb',
}

app = Flask(__name__)

model_file_path = '/home/Flask_Searver/tfidf_vectorizer_model.pkl'
matrix_file_path = "/home/Flask_Searver/document_matrix.npz"

# 모델 불러오기
with open(model_file_path, 'rb') as model_file:
    loaded_vectorizer = pickle.load(model_file)

# 행렬 불러오기
loaded_document_matrix = load_npz(matrix_file_path)

# 추천 함수 정의
def get_recommendations(cosine_sim, df):
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:10]
    book_indices = [i[0] for i in sim_scores]
    selected_book_ids = df[df['book_id'].isin(book_indices)]['book_id'].to_list()

    return selected_book_ids


# 기본 쿼리
def execute_dynamic_query(topic, genre, category):
    base_query = """
        SELECT DISTINCT b.book_id
        FROM book b
    """

    # 동적으로 조건 추가
    if topic:
        topic_values = ', '.join([f"'{t.strip()}'" for t in topic.split(',')])
        base_query += f" AND b.topic IN ({topic_values})"
    if genre:
        genre_values = ', '.join([f"'{g.strip()}'" for g in genre.split(',')])
        base_query += f" AND b.genre IN ({genre_values})"
    if category:
        category_values = ', '.join([f"'{c.strip()}'" for c in category.split(',')])
        base_query += f" AND b.category IN ({category_values})"

    # LIMIT 추가
    # base_query += " LIMIT 100"

    # 생성된 쿼리 출력 (디버깅용)
    # print("Generated Query:", base_query)

    return base_query


# 문장에서 불필요한 부분 제거
def preprocess_sentence(sentence, name=''):
    if not isinstance(sentence, str):
        print('문자열이 아닙니다.')
        return sentence
    name_parts = name.split(' ')
    if len(name_parts) > 1:
        sentence = re.sub(name_parts[0], '', sentence)
        sentence = re.sub(name_parts[1], '', sentence)

    sentence = re.sub('[^가-힣a-zA-Z0-9]+|[ㄱ-ㅎㅏ-ㅣ]', ' ', sentence)

    return sentence




@app.route('/api/list', methods=['GET'])
def list_books():
    start = time.time()

    keyword = urllib.parse.unquote(request.args.get('keyword'))
    topic = urllib.parse.unquote(request.args.get('topic'))
    genre = urllib.parse.unquote(request.args.get('genre'))
    category = urllib.parse.unquote(request.args.get('category'))


    ##### DB

    try:
        conn = pymysql.connect(**config)

        cursor = conn.cursor()

        print('★☆☆☆ connect 성공')

        qurey = execute_dynamic_query(topic, genre, category)

        cursor.execute(qurey)

        print("★★☆☆ 쿼리 성공")

        result = [i[0] for i in cursor.fetchall()]

        print("★★★☆ 쿼리 결과 성공")
        

        # print(book_id_df)
    except:
        print("☆☆☆☆ mysql db 연결 실패")

    try:
        book_id_df = pd.DataFrame({'book_id': result})

    except:
        print("☆☆☆☆ 데이터 프레임 화 실패")
    else:
        print("★★★★ 데이터 프레임 화 성공")
    ##### DB

        cbf_start = time.time()
        ##### CBF
        kiwi = Kiwi()

        preprocessed_keyword = preprocess_sentence(keyword)

        if preprocessed_keyword:
            search_tokens = kiwi.analyze(preprocessed_keyword)[0][0]
            search_keywords = [token[0] for token in search_tokens]

            search_vector = loaded_vectorizer.transform([' '.join(search_keywords)])
            print("◎◎◎◎◎ 백터화 성공")


            cosine_sim = cosine_similarity(search_vector, loaded_document_matrix)
            print("◎◎◎◎◎ 코사인 성공")

            result = get_recommendations(cosine_sim, book_id_df)
            print("◎◎◎◎◎ 추천 성공")

            end = time.time()
            cbf_end = time.time()

            print("CBF 걸린 시간 : ", cbf_end - cbf_start)

            print("걸린 시간 : ", end - start)
            return jsonify({'result': result})
        ##### CBF


    response_data = {'message': 'Hello from Flask! This is the response from /list endpoint.'}
    return response_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
