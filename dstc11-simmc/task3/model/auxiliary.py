from torch import nn


class BoxEmbedding(nn.Module):
    ''' 用于编码BOX Position Information'''
    def __init__(self, hidden_dim):
        super(BoxEmbedding, self).__init__()
        self.box_linear = nn.Linear(6, hidden_dim)
        self.box_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, box_feat):
        box_feat = self.box_linear(box_feat)
        transformed_box = self.box_layer_norm(box_feat)
        return transformed_box


class NoCorefHead(nn.Module):
    ''' 用于进行指代消解的Head: 感觉还是一个二分类的loss'''

    def __init__(self, hidden_dim):
        super(NoCorefHead, self).__init__()
        self.no_coref_linear = nn.Linear(hidden_dim, 2)

    def forward(self, no_coref_vector):
        coref_cls = self.no_coref_linear(no_coref_vector)
        return coref_cls


class DisambiguationHead(nn.Module):
    ''' 用于进行消歧检测的Head'''

    def __init__(self, hidden_dim):
        super(DisambiguationHead, self).__init__()
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.linear(x)


class FashionEncoderHead(nn.Module):
    def __init__(self, hidden_dim):
        ''' 用于将特征的输出进行分类： 时尚类信息 多标签2分类'''
        super(FashionEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        # self.aggregator_layer_norm = nn.LayerNorm(2*hidden_dim)

        self.coref_linear = nn.Linear(2*hidden_dim, 2)  # 针对当前的Object是否被提及的Loss

        self.size_linear = nn.Linear(2*hidden_dim, 6)
        self.available_sizes_linear = nn.Linear(2*hidden_dim, 6)  # sigmoid is applied later by 因为AvaliableSize的Loss不太一样
        self.brand_linear = nn.Linear(2*hidden_dim, 26)
        self.color_linear = nn.Linear(2*hidden_dim, 71)
        self.pattern_linear = nn.Linear(2*hidden_dim, 36)
        self.sleeve_length_linear = nn.Linear(2*hidden_dim, 6)
        self.asset_type_linear = nn.Linear(2*hidden_dim, 12)
        self.type_linear = nn.Linear(2*hidden_dim, 18)
        self.price_linear = nn.Linear(2*hidden_dim, 45)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 26)

    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)

        coref = self.coref_linear(aggregated)
        size = self.size_linear(aggregated)
        available_sizes = self.available_sizes_linear(aggregated)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        pattern = self.pattern_linear(aggregated)
        sleeve_length = self.sleeve_length_linear(aggregated)
        asset_type = self.asset_type_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)

        return coref, size, available_sizes, brand, color, pattern, sleeve_length, asset_type, type_, price, customer_review


class FurnitureEncoderHead(nn.Module):
    ''' 用于将家具类信息进行输出进行分类的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureEncoderHead, self).__init__()
        self.aggregator = nn.Linear(2*hidden_dim, 2*hidden_dim)
        # self.aggregator_layer_norm = nn.LayerNorm(2*hidden_dim)

        self.coref_linear = nn.Linear(2*hidden_dim, 2)

        self.brand_linear = nn.Linear(2*hidden_dim, 12)
        self.color_linear = nn.Linear(2*hidden_dim, 9)
        self.materials_linear = nn.Linear(2*hidden_dim, 7)
        self.type_linear = nn.Linear(2*hidden_dim, 10)
        self.price_linear = nn.Linear(2*hidden_dim, 10)
        self.customer_review_linear = nn.Linear(2*hidden_dim, 19)

    def forward(self, concat_vector):
        ''' concat_vector: concat of obj_index_vector and st_vector '''
        aggregated = self.aggregator(concat_vector)

        coref = self.coref_linear(aggregated)
        brand = self.brand_linear(aggregated)
        color = self.color_linear(aggregated)
        materials = self.materials_linear(aggregated)
        type_ = self.type_linear(aggregated)
        price = self.price_linear(aggregated)
        customer_review = self.customer_review_linear(aggregated)
        return coref, brand, color, materials, type_, price, customer_review


class IntentHead(nn.Module):
    ''' 用于进行消歧检测的Head'''

    def __init__(self, hidden_dim):
        super(IntentHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class IntentSubHead(nn.Module):
    ''' 用于进行消歧检测的Sub Head'''

    def __init__(self, hidden_dim):
        super(IntentSubHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class DisamHead(nn.Module):
    ''' 用于进行消歧检测的Head'''

    def __init__(self, hidden_dim):
        super(DisamHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class DisamAllHead(nn.Module):
    ''' 用于学习disambiguation 场景下的是否存在all 的信息'''

    def __init__(self, hidden_dim):
        super(DisamAllHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.cls(self.aggregater(x))
    


class DisamTypeHead(nn.Module):
    ''' 
        用于学习disambiguation 场景下的type 信息
        使用27是因为准备将0作为空值进行处理
    '''

    def __init__(self, hidden_dim):
        super(DisamTypeHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 27)

    def forward(self, x):
        return self.cls(self.aggregater(x))
    
    
class FashionTypeHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionTypeHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 18)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionPriceHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionPriceHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 48)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionCustomerReviewHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionCustomerReviewHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 28)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionBrandHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionBrandHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 27)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionSizeHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionSizeHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionPatternHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionPatternHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 34)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionColorHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionColorHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 69)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionSleeveLengthHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionSleeveLengthHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FashionAvaliableSizeHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FashionAvaliableSizeHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurnitureTypeHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureTypeHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 11)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurnitureMaterialHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureMaterialHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 7)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurniturePriceHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurniturePriceHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 14)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurnitureBrandHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureBrandHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 13)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurnitureCustomerRatingHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureCustomerRatingHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 21)

    def forward(self, x):
        return self.cls(self.aggregater(x))


class FurnitureColorHead(nn.Module):
    ''' 用于学习slot-value的Head'''

    def __init__(self, hidden_dim):
        super(FurnitureColorHead, self).__init__()
        self.aggregater = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        return self.cls(self.aggregater(x))