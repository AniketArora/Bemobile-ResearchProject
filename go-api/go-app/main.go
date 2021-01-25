package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	tg "github.com/galeone/tfgo"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	// "github.com/tensorflow/tensorflow/tensorflow/go/op"
)

var apiVersion = "/api/v1/"

func headerHelper(w http.ResponseWriter, contentType string) {
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Access-Control-Allow-Origin", "*")
}

func helloWorld(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	headerHelper(w, "application/json")
	message := "Hello, World!"
	json.NewEncoder(w).Encode(message)
	fmt.Println("Ran: Hello, World!")
}

func predictTraffic(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Started prediction")
	startPrediction := time.Now()

	headerHelper(w, "application/json")

	startLoad := time.Now()
	fmt.Println("Started loading model")
	tfModel := tg.LoadModel("model/BeMobileModel", []string{"serve"}, nil)
	elapsedLoad := time.Since(startLoad)
	fmt.Println("Finished loading model, took", elapsedLoad)

	var predReqData struct {
		Conv [][][]float32
		Val  [][]float32
	}

	newerr := json.NewDecoder(r.Body).Decode(&predReqData)

	if newerr != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Printf("Error parsing request body: %v", newerr.Error())
		json.NewEncoder(w).Encode(newerr.Error())
		return
	}

	Conv, err := tf.NewTensor([1][60][30][1]float32{})

	if err != nil {
		fmt.Printf("Error input model: %s\n", err.Error())
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(newerr.Error())
		return
	}

	Val, err := tf.NewTensor([1][9]float32{})

	if err != nil {
		fmt.Printf("Error input model: %s\n", err.Error())
		w.WriteHeader(500)
		json.NewEncoder(w).Encode(newerr.Error())
		return
	}

	fmt.Println("No error input model")

	results := tfModel.Exec([]tf.Output{
		tfModel.Op("StatefulPartitionedCall", 0),
	}, map[tf.Output]*tf.Tensor{
		tfModel.Op("serving_default_input_1", 0): Conv,
		tfModel.Op("serving_default_input_2", 0): Val,
	})

	predictions := results[0]
	w.WriteHeader(200)
	json.NewEncoder(w).Encode(predictions.Value())

	elapsedPrediction := time.Since(startPrediction)
	fmt.Println("Finished prediction, took", elapsedPrediction)
	return
}

func endpoints() {
	http.HandleFunc(apiVersion, helloWorld)
	http.HandleFunc(apiVersion+"predict/", predictTraffic)
}

func start() {
	startAPIServer := time.Now()
	fmt.Println("Starting Api Server")
	endpoints()
	elapsedTimeAPIServer := time.Since(startAPIServer)
	fmt.Println("Started Api Server, took", elapsedTimeAPIServer)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Printf("Error in Api server: %s\n", err.Error())
	}
}

func main() {
	start()
	// run()
}