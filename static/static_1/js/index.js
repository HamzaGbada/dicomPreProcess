
$(document).ready(function() {


    var element = $('#dicomImage').get(0);
    cornerstone.enable(element);
    // Listen for the change event on our input element so we can get the
    // file selected by the user
    $('#selectFile').on('change', function(e) {
        // Add the file to the cornerstoneFileImageLoader and get unique
        // number for that file
        var file = e.target.files[0];
        var index = cornerstoneFileImageLoader.addFile(file);
        // create an imageId for this image
        var imageId = "dicomfile://" + index;
        // load and display this image into the element
        var element = $('#dicomImage').get(0);

        cornerstone.loadImage(imageId).then(function(image) {
            cornerstone.displayImage(element, image);

        });
        cornerstone.loadImage(imageId).then(function (image) {
            var imageId1 = image.imageId;
            var minPixelValue = image.minPixelValue;
            var maxPixelValue = image.maxPixelValue;
            var pixelData = image.getPixelData();
            const M = 3;        // Note 2nd dimention is not relevant here

            var arr = [];
            for (var i = 0; i < M; i++) {
                arr[i] = [i+6,i*2,i];
            }
            console.log(arr);
            // document.getElementById("gammaButton").onclick = function(){
            //     var gamma = document.getElementById("gammaValue").value;
            //     // var url = 'https://jsonplaceholder.typicode.com/posts';
            //     var url = 'http://127.0.0.1:5000/gammaCorrection/'+gamma;
            //     // var url = 'http://127.0.0.1:5000/dicomImage/'
            //     fetch(url, {
            //         method: 'POST',
            //         body: JSON.stringify({
            //             width: 512,
            //             height: 512,
            //             pixel_data: pixelData,
            //         }),
            //         headers: {
            //             'Content-type': 'text/plain; charset=UTF-8'
            //             // 'Access-Control-Allow-Origin': POST ,
            //             // 'Access-Control-Allow-Headers': 'Content-Type'
            //         },
            //     })
            //         .then((response) => response.json())
            //         .then((json) => console.log(json));
            // }
            console.log("image ID");
            console.log(imageId1);
            console.log("min Pixel Value");
            console.log(minPixelValue);
            console.log("Max Pixel Value");
            console.log(maxPixelValue);
            console.log("Pixel Data ");
            console.log(pixelData);
        });



    });

});
$(document).ready(function() {

    var element = $('#dicomImage1').get(0);
    cornerstone.enable(element);

    // Listen for the change event on our input element so we can get the
    // file selected by the user
    $('#selectFile1').on('change', function(e) {
        // Add the file to the cornerstoneFileImageLoader and get unique
        // number for that file
        var file = e.target.files[0];
        var index = cornerstoneFileImageLoader.addFile(file);
        // create an imageId for this image
        var imageId = "dicomfile://" + index;
        // load and display this image into the element
        var element = $('#dicomImage1').get(0);
        cornerstone.loadImage(imageId).then(function(image) {
            cornerstone.displayImage(element, image);
        });

    });

});

document.getElementById("gammaButton").onclick = function(){
    var gamma = document.getElementById("gammaValue").value;
    document.getElementById('form_id').action = 'http://127.0.0.1:5000/gammaCorrection/'+gamma;
}
document.getElementById("otsuButton").onclick = function(){
    var otsu = document.getElementById("otsuValue").value;
    document.getElementById('form_id').action = 'http://127.0.0.1:5000/otsuThreshold/'+otsu;
}
document.getElementById("contrastButton").onclick = function(){
    var contrast = document.getElementById("contrastValue").value;
    var brightness = document.getElementById("brightnessValue").value;
    document.getElementById('form_id').action = 'http://127.0.0.1:5000/contrastAdjust/'+contrast+'/'+brightness;
}
