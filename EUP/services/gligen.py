class GligenService():

    def extractGligen(self, positive, negative):
        gligen_pos = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in positive
        ]
        gligen_neg = [
            c[1]['gligen'] if 'gligen' in c[1] else None
            for c in negative
        ]
        return gligen_pos, gligen_neg
    
    def sliceGligen(self, tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen):
        tile_h_end = tile_h + tile_h_len
        tile_w_end = tile_w + tile_w_len
        if gligen is None:
            return
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        gligen_areas = gligen[2]
        
        gligen_areas_new = []
        for emb, h_len, w_len, h, w in gligen_areas:
            h_end = h + h_len
            w_end = w + w_len
            if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
                new_h = max(0, h - tile_h)
                new_w = max(0, w - tile_w)
                new_h_end = min(tile_h_end, h_end - tile_h)
                new_w_end = min(tile_w_end, w_end - tile_w)
                gligen_areas_new.append((emb, new_h_end - new_h, new_w_end - new_w, new_h, new_w))

        if len(gligen_areas_new) == 0:
            del cond['gligen']
        else:
            cond['gligen'] = (gligen_type, gligen_model, gligen_areas_new)

    def prepsreSlicedGligen(self, pos, neg, gligen_pos, gligen_neg, tile_h, tile_h_len, tile_w, tile_w_len):
        # Modify 'pos' and 'neg' lists in place by slicing the gligen conditions
        for cond, gligen in zip(pos, gligen_pos):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
        for cond, gligen in zip(neg, gligen_neg):
            self.sliceGligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
        
        # Return the modified 'pos' and 'neg' lists
        return pos, neg