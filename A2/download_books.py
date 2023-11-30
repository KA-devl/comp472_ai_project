# grabbed from https://superfastpython.com/threadpoolexecutor-download-books/#How_to_Create_a_Pool_of_Worker_Threads_with_ThreadPoolExecutor
import itertools
import re
from os.path import join
from os import makedirs
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from bs4 import BeautifulSoup

BOOKS_TO_DOWNLOAD = 100

# download html and return parsed doc or None on error
def download_url(urlpath):
    try:
        # open a connection to the server
        with urlopen(urlpath, timeout=3) as connection:
            # read the contents of the html doc
            return connection.read()
    except:
        # bad url, socket timeout, http forbidden, etc.
        return None


# decode downloaded html and extract all <a href=""> links
def get_urls_from_html(content):
    # decode the provided content as ascii text
    html = content.decode('utf-8')
    # parse the document as best we can
    soup = BeautifulSoup(html, 'html.parser')
    # find all all of the <a href=""> tags in the document
    atags = soup.find_all('a')
    # get all links from a tags
    return [tag.get('href') for tag in atags]


# return all book unique identifiers from a list of raw links
def get_book_identifiers(links):
    # define a url pattern we are looking for
    pattern = re.compile('/ebooks/[0-9]+')
    # process the list of links for those that match the pattern
    books = set()
    for link in links:
        # check of the link matches the pattern
        if not pattern.match(link):
            continue
        # extract the book id from /ebooks/nnn
        book_id = link[8:]
        # store in the set, only keep unique ids
        books.add(book_id)
    return books


# download one book from project gutenberg
def download_book(book_id, save_path):
    # construct the download url
    url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
    # download the content
    data = download_url(url)
    if data is None:
        return f'Failed to download {url}'
    # create local path
    save_file = join(save_path, f'{book_id}.txt')
    # save book to file
    with open(save_file, 'wb') as file:
        file.write(data)
    return f'Saved {save_file}'


# download all top books from project gutenberg
def download_all_books(url, save_path):
    data = download_url(url)
    print(f'.downloaded {url}')
    links = get_urls_from_html(data)
    print(f'.found {len(links)} links on the page')
    book_ids = get_book_identifiers(links)
    print(f'.found {len(book_ids)} unique book ids')
    makedirs(save_path, exist_ok=True)
    with ThreadPoolExecutor(min(len(book_ids), BOOKS_TO_DOWNLOAD)) as exe:
        futures = [exe.submit(download_book, book, save_path) for book in itertools.islice(book_ids, BOOKS_TO_DOWNLOAD)]
        for future in as_completed(futures):
            print(future.result())


# entry point
URL = 'https://www.gutenberg.org/browse/scores/top'
DIR = 'books'
# download top books
download_all_books(URL, DIR)
