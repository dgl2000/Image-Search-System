$(document).on('click', 'a.images-to-search', function(e) {
        e.preventDefault(); // stop anchor navigation
        var id = $(this).attr('id')
        console.log(id);
        $.ajax({
            url: 'http://127.0.0.1:5000/image_id/' + id,
            type: 'POST',
            data: id,
            cache: false,
            processData: false,
            contentType: false,
            success: function (data) {
                console.log(data)
                window.location = 'image-search.html';
            },
            error: function (error) {
                console.log(error);
            },
        })
});

$(document).on('ready', function () {
    $(".index-title").text("Oxford Buildings");
    var dir = "img/oxbuild_images_100/";
    $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/show_all_image/',
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
                                    <h2>Click to Search: ${filename}</h2>
                                    <a class="images-to-search" id=${filename} href="image-search.html">Search Similar Images</a>
                                </figcaption>                    
                            </figure>
                        </div>
                    `
            });
            $(".gallery-image").html(content);
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
                // alert("Oops, something goes wrong with this page...");
                // window.location = 'my_module.html';
        },
    })
});


$('#search-btn-submit').click(function () {
    var dir = "img/oxbuild_images_100/";
    var search_key = $('#search-btn-content').val();
    if (search_key.length!==0){
        console.log(search_key.length!==0);
            $.ajax({
            //This will retrieve the contents of the folder if the folder is configured as 'browsable'
            url: 'http://127.0.0.1:5000/search_images/' + search_key,
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
                                        <h2>Click to Search: ${filename}</h2>
                                        <a class="images-to-search" id=${filename} href="image-search.html">Search Similar Images</a>
                                    </figcaption>                    
                                </figure>
                            </div>
                        `
                    });
                $(".gallery-image").html(content);
                $(".index-title").text("Filtered Images");
            },
            error: function (error) {
                    console.log(error);
                    console.log(error.responsehtml);
                    // alert("Oops, something goes wrong with this page...");
                    // window.location = 'my_module.html';
            },
        })
    }
    else{
        $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/show_all_image/',
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
                                    <h2>Click to Search: ${filename}</h2>
                                    <a class="images-to-search" id=${filename} href="image-search.html">Search Similar Images</a>
                                </figcaption>                    
                            </figure>
                        </div>
                    `
            });
            $(".gallery-image").html(content);
        },
        error: function (error) {
                console.log(error);
                console.log(error.responsehtml);
                // alert("Oops, something goes wrong with this page...");
                // window.location = 'my_module.html';
        },
    })
  }
});

$('#image-search-btn-submit').click(function(){
    var dir = "img/oxbuild_images_100/";
    let filepath = prompt("Please enter the image file path or url:", "http://img_gallery/img_0001.jpg");
    if (filepath == null || filepath === "") {
        alert("No filepath...")
      } else {
        alert("Searching...Please wait :)")
        $.ajax({
        //This will retrieve the contents of the folder if the folder is configured as 'browsable'
        url: 'http://127.0.0.1:5000/input_image_search/'+ filepath,
        type: 'GET',
        success: function (data) {
            var content = ""
            $.each(data, function (index, value) {
                $.each(value.imgs, function (index2, filename) {
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
            $(".gallery-image").html(content);
            $(".index-title").text("Search Results");
            alert("Results below ↓");
        },
        error: function (error) {
            console.log(error);
            alert("Something went wrong...")
        },
    })
    }

});