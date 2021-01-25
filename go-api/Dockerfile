#Installing tensorflow 2.4.0 with go installed
FROM aroraaniket/tensorflow2-go:latest

#Installing tfgo, a tensorflow library that makes it easier to use tensorflow
RUN go get github.com/galeone/tfgo

WORKDIR /go-app
COPY ./go-app .

RUN go build -o main .

EXPOSE 8080
CMD ["./main"]