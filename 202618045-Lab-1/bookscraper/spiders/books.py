import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com/"]

    page_count = 1
    max_pages = 5

    def parse(self, response):

        for book in response.css("article.product_pod"):

            book_data = {
                "title": book.css("h3 a::attr(title)").get(),
                "price": book.css("p.price_color::text").get(),
                "availability": "".join(
                    book.css("p.instock.availability::text").getall()
                ).strip(),
                "rating": book.css("p.star-rating::attr(class)").get().replace("star-rating ", "")
            }

            book_url = response.urljoin(
                book.css("h3 a::attr(href)").get()
            )

            yield response.follow(
                book_url,
                callback=self.parse_book,
                meta={"book_data": book_data}
            )

        if self.page_count < self.max_pages:
            next_page = response.css("li.next a::attr(href)").get()

            if next_page:
                self.page_count += 1
                yield response.follow(next_page, callback=self.parse)

    def parse_book(self, response):

        book_data = response.meta["book_data"]

        book_data["upc"] = response.css("table tr:nth-child(1) td::text").get()

        book_data["category"] = response.css(
            "ul.breadcrumb li:nth-child(3) a::text"
        ).get()

        book_data["description"] = response.css(
            "#product_description + p::text"
        ).get()

        book_data["number_of_reviews"] = response.css(
            "table tr:nth-child(7) td::text"
        ).get()

        book_data["product_url"] = response.url

        yield book_data