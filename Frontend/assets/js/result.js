// document.getElementsByClassName('preloader-enc')[0].style.display='block';
// document.getElementsByClassName('main-content')[0].style.display='none';
document.getElementsByClassName('close')[0].addEventListener('click',()=>{document.getElementsByClassName('error')[0].style.top='-250px'})

let urlParams = new URLSearchParams(location.search);
let id=urlParams.get('item').split('/')[5];
console.log(id);
console.log(urlParams.get('item').split('/'))
var overall;

let xhttp=new XMLHttpRequest();
xhttp.open('GET',`https://protected-cliffs-92683.herokuapp.com/details?id=${id}`,false)
xhttp.onreadystatechange = () => {
    if(xhttp.readyState==4 && xhttp.status == 200) {
        try {
            let x = JSON.parse(xhttp.responseText);
            overall=x;
            console.log(overall)
            document.getElementsByClassName('preloader-enc')[0].style.display='none';
            document.getElementsByClassName('main-content')[0].style.display='block';
            document.getElementsByClassName('item-name')[0].innerHTML=x.name;
            document.getElementById('product-link').href=x.url;
            document.getElementsByClassName('trating')[0].style.width=`${x.ratingold*20}%`;
            document.getElementById('trating-tool').title=`${x.ratingold*20}%`;
            document.getElementsByClassName('orating')[0].style.width=`${x.newrating*20}%`;
            document.getElementById('orating-tool').title=`${x.newrating*20}%`;
            document.getElementsByClassName('item-seller')[0]=x.price;
            switchSome(0);
        } catch(err){ 
            console.log(err);
            document.getElementsByClassName('error')[0].innerHTML=`
                <span class="close">&times</span>
                An error occured. Please try reloading the page.
            `
            document.getElementsByClassName('error')[0].style.top='0px';
            document.getElementsByClassName('preloader-enc')[0].style.display='none';
            setTimeout(()=> {document.getElementsByClassName('error')[0].style.top='-250px'}, 2000)
            }
    } else if(xhttp.readyState==4 && xhttp.status != 200) {
        document.getElementsByClassName('preloader-enc')[0].style.display='none';
        document.getElementsByClassName('error')[0].innerHTML=`
        <span class="close">&times</span>
        An error occured. Please try reloading the page.
    `
        document.getElementsByClassName('error')[0].style.top='0px';
        document.getElementsByClassName('preloader-enc')[0].style.display='none';
        setTimeout(()=> {document.getElementsByClassName('error')[0].style.top='-250px'}, 2000)
        }
}
xhttp.send();


function switchSome(inte) {
    for(let i=0;i<2;i++) {
        document.getElementsByClassName('nav-link')[i].classList.remove('active')
    }
    document.getElementsByClassName('nav-link')[inte].classList.add('active');
    if (inte==0) {
        document.getElementsByClassName('reviews')[0].innerHTML = '';
        console.log(overall);
        for(let i=0;i<overall.real.length;i++) {
            document.getElementsByClassName('reviews')[0].innerHTML+=`
            <div class="card">
                <div class="card-header">
                    ${overall.real[i].review_author}
                </div>
                <div class="card-body review">
                    <div><img src="assets/images/product.png" class="review-img hide-image"></div>
                    <div class="review-text">${overall.real[i].review_text}</div>
                </div>
            </div>  
            `;
        }
        if(overall.real.length==0) {
            document.getElementsByClassName('reviews')[0].innerHTML+=`
            <div class="card">
                <div class="card-header">
                   No real reviews amongst the reviews scraped.
                </div>
            </div>  
            `;
        }
    } else if (inte==1) {
        document.getElementsByClassName('reviews')[0].innerHTML = '';
        for(let i=0;i<overall.fake.length;i++) {
            document.getElementsByClassName('reviews')[0].innerHTML+=`
            <div class="card">
                <div class="card-header">
                    ${overall.fake[i].review_author}
                </div>
                <div class="card-body review">
                    <div><img src="assets/images/product.png" class="review-img hide-image"></div>
                    <div class="review-text">${overall.fake[i].review_text}</div>
                </div>
            </div>  
            `;
        }
        if(overall.fake.length==0) {
            document.getElementsByClassName('reviews')[0].innerHTML+=`
            <div class="card">
                <div class="card-header">
                   No fake reviews amongst the reviews scraped.
                </div>
            </div>  
            `;
        }
    }
    window.scrollY = window.scrollY+100
}