$(document).on('ready', function () {
    var dir = "img/oxbuild_images_100/";
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/query_image_id/',
        type: 'GET',
        success: function (data) {
            console.log(data)
            $('#image-filename').html(data)
            $('.query-image').html(`<img src=${dir + data} alt="Image" class="img-fluid">`)

        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
        },
    })
});

$('#image-search-btn').click(function () {
    var dir = "img/oxbuild_images_100/";
    console.log("hello");
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/query_similar_image_cnn/',
        type: 'GET',
        success: function (data) {
            var content = ""
            $.each(data, function (index, value) {
                $.each(value.imgs, function (index2, filename) {
                    console.log(filename)
                    content +=
                    `
                      <div class="col-xl-3 col-lg-4 col-md-6 col-sm-6 col-12 mb-5">
                        <figure class="effect-ming tm-video-item">
                            <img src=${dir + filename} alt="Image" class="img-fluid">
                            <figcaption class="d-flex align-items-center justify-content-center">
                                <h2>${filename}</h2>
                            </figcaption>                    
                        </figure>
                        <div class="d-flex justify-content-center tm-text-gray">
                            <span class="tm-text-gray-dark">Rate=${value.vals[index2]}</span>
                        </div>
                    </div>
                    `
                });
            });
            $(".search-result-title").html(`
                <h2 class="col-12 tm-text-primary">
                    Search Results
                </h2>`
            );
            $(".search-results").html(content);
            $('html,body').animate({scrollTop: $(".search-result-title").offset().top}, 'slow');
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
                // alert("Oops, something goes wrong with this page...");
                // window.location = 'my_module.html';
        },
    })
});

$('#image-search-100-btn').click(function () {
    var dir = "img/oxbuild_images_100/";
    console.log("100 btn");
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/query_similar_image_cnn100/',
        type: 'GET',
        success: function (data) {
            var content = ""
            $.each(data, function (index, value) {
                $.each(value.imgs, function (index2, filename) {
                    console.log(filename)
                    content +=
                    `
                      <div class="col-xl-3 col-lg-4 col-md-6 col-sm-6 col-12 mb-5">
                        <figure class="effect-ming tm-video-item">
                            <img src=${dir + filename} alt="Image" class="img-fluid">
                            <figcaption class="d-flex align-items-center justify-content-center">
                                <h2>${filename}</h2>
                            </figcaption>                    
                        </figure>
                        <div class="d-flex justify-content-center tm-text-gray">
                            <span class="tm-text-gray-dark">Rate=${value.vals[index2]}</span>
                        </div>
                    </div>
                    `
                });
            });
            $(".search-result-title").html(`
                <h2 class="col-12 tm-text-primary">
                    Search Results
                </h2>`
            );
            $(".search-results").html(content);
            $('html,body').animate({scrollTop: $(".search-result-title").offset().top}, 'slow');
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
        },
    })
});



$('#image-search-sift-btn').click(function () {
    var dir = "img/oxbuild_images_100/";
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/query_similar_image_sift/',
        type: 'GET',
        success: function (data) {
            var content = ""
            $.each(data, function (index, value) {
                if (index!==0){
                    content +=
                    `
                      <div class="col-xl-3 col-lg-4 col-md-6 col-sm-6 col-12 mb-5">
                        <figure class="effect-ming tm-video-item">
                            <img src=${dir + value['image_name']} alt="Image" class="img-fluid">
                            <figcaption class="d-flex align-items-center justify-content-center">
                                <h2>${value['image_name']}</h2>
                            </figcaption>                    
                        </figure>
                        <div class="d-flex justify-content-center tm-text-gray">
                            <span class="tm-text-gray-dark">Distance=${value['dist']}</span>
                        </div>
                    </div>
                    `
                }
            });
            $(".search-result-title").html(`
                <h2 class="col-12 tm-text-primary">
                    Search Results
                </h2>`
            );
            $(".search-results").html(content);
            $('html,body').animate({scrollTop: $(".search-result-title").offset().top}, 'slow');
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
        },
    })
});


$('#image-search-encoder-btn').click(function () {
    var dir = "img/oxbuild_images_100/";
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/query_similar_image_encoder/',
        type: 'GET',
        success: function (data) {
            var content = ""
            $.each(data, function (index, filename) {
                content +=
                `
                  <div class="col-xl-3 col-lg-4 col-md-6 col-sm-6 col-12 mb-5">
                    <figure class="effect-ming tm-video-item">
                        <img src=${dir + filename} alt="Image" class="img-fluid">
                        <figcaption class="d-flex align-items-center justify-content-center">
                            <h2>${filename}</h2>
                        </figcaption>                    
                    </figure>
                </div>
                `
            });
            $(".search-result-title").html(`
                <h2 class="col-12 tm-text-primary">
                    Search Results
                </h2>`
            );
            $(".search-results").html(content);
            $('html,body').animate({scrollTop: $(".search-result-title").offset().top}, 'slow');
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
        },
    })
});